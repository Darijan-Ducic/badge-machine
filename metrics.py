"""Chance-aware VTT metrics and the turns-per-arm power calculation, for labs that judge changes on small paired turn
sets. Additive to experiment.py (word_acc, viseme_acc are unchanged there). Guide: .claude/skills/iterate/references/
metrics.md. Every function takes VTT texts and STT texts in turn order; "" is an empty read.

Pooled metrics (one number over a turn set; turns whose STT has no units add nothing):
  phoneme_acc      1 - phoneme edits / STT phonemes (CMUdict first pronunciation, stress stripped; an
                   out-of-vocabulary word is one token that matches only the same word). Can go below 0.
  viseme_acc_norm  1 - viseme edits / max(read visemes, STT visemes), unit costs, summed over turns. In [0, 1].
  content_recall   STT content words (not in STOPWORDS; counts clipped) found in the read / STT content words.
  keyword_hit      share of turns whose read contains at least one of the STT's content words.

Impostor-corrected metrics (need the whole turn set; a constant read scores chance exactly):
  viseme_lift      per turn: viseme acc vs its own STT minus the mean viseme acc vs every other turn's STT;
                   the unweighted mean over turns. Any constant read scores exactly 0. A read of the right LENGTH
                   still earns lift (a filler of the STT's word count: +0.175 on 41 obama turns); pass nearest=k
                   to compare only against the k impostors closest in STT viseme count.
  sentence_id      closed set: rank the turn's own STT among the candidate STTs (default: the turns' STTs) by
                   unit-cost viseme edits / max length to the read. top1 = share where it is nearest (ties share
                   credit), chance 1/N; percentile = share of impostors it beats (ties count half), chance 0.5.

Power (paired design, two-sided):
  paired_power / mde_paired / turns_needed   paired t test via the noncentral t; `sd` is the SD of the per-turn
                   difference (cand - base), `effect` the true mean difference.
  sign_power / sign_turns_needed   exact sign test on m non-tied turns when each turn goes up with probability p_up.
  split_half_sd    the null SD of a per-turn difference, from each turn's K stored samples: every split into
                   floor(K/2) | rest, each half read by `pick` (default the most common text, ties to the first).
"""

from collections import Counter
from itertools import combinations
from math import comb

import cmudict
import numpy as np
from scipy import optimize, stats

import experiment
import visemes
from preheat import _normalize

# English function words: pronouns, articles, auxiliaries, prepositions, conjunctions, common adverbs.
STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being below between both
but by can could did do does doing down during each else ever few for from further had has have having he her here
hers herself him himself his how i if in into is it it's its itself just let's me more most my myself no nor not
now of off on once only or other our ours ourselves out over own same she should so some such than that that's the
their theirs them themselves then there there's these they they're this those through to too under until up us very
was we we'll we're we've were what when where which while who whom why will with would you you're your yours
yourself yourselves i'm i've i'll don't didn't doesn't isn't aren't wasn't weren't won't can't couldn't wouldn't
shouldn't oh uh um yeah ok okay
""".split())


# ---------------------------------------------------------------- pooled metrics


def _phonemes(text: str) -> list[str]:
    out = []
    for word in visemes._words(text):
        prons = visemes._lexicon().get(word)
        out += [p.rstrip("012") for p in prons[0]] if prons else [f"<{word}>"]
    return out


def phoneme_acc(vtts: list[str], stts: list[str]) -> float | None:
    """1 - phoneme edits over the STT phoneme count, summed over turns; None without STT phonemes."""
    return experiment._accuracy([(_phonemes(v), _phonemes(s)) for v, s in zip(vtts, stts)])


def viseme_acc_norm(vtts: list[str], stts: list[str]) -> float | None:
    """1 - viseme edits over max(read, STT) viseme counts, summed over turns; None when every turn is empty."""
    pairs = [(visemes.to_visemes(v), visemes.to_visemes(s)) for v, s in zip(vtts, stts)]
    total = sum(max(len(a), len(b)) for a, b in pairs)
    return 1 - sum(experiment._edits(a, b) for a, b in pairs) / total if total else None


def _content(text: str) -> Counter:
    return Counter(w for w in _normalize(text).split() if w not in STOPWORDS)


def content_recall(vtts: list[str], stts: list[str]) -> float | None:
    """STT content words recovered by the read (clipped counts) over all STT content words; None without any."""
    pairs = [(_content(v), _content(s)) for v, s in zip(vtts, stts)]
    total = sum(sum(said.values()) for _, said in pairs)
    return sum(sum((read & said).values()) for read, said in pairs) / total if total else None


def keyword_hit(vtts: list[str], stts: list[str]) -> float | None:
    """Share of turns (with STT content words) whose read has at least one of them; None without such turns."""
    hits = [bool(_content(v) & said) for v, said in ((v, _content(s)) for v, s in zip(vtts, stts)) if said]
    return sum(hits) / len(hits) if hits else None


def per_turn_word_acc(read: str, stt: str) -> float | None:
    return experiment.word_acc([read], [stt])


def per_turn_viseme_acc(read: str, stt: str) -> float | None:
    return experiment.viseme_acc([read], [stt])


# ---------------------------------------------------------------- impostor-corrected metrics


def viseme_lift_turns(vtts: list[str], stts: list[str], nearest: int | None = None) -> list[float]:
    """Per turn: viseme acc vs its own STT minus the mean vs impostor STTs: every other turn's, or with `nearest`
    = k only the k other turns whose STT viseme count is closest to its own (ties: earlier turn first), so a read
    earns nothing from its length alone. Needs >= 2 turns."""
    if len(stts) < 2:
        raise ValueError("viseme_lift needs at least 2 turns: the other turns are the impostors")
    saids = [visemes.to_visemes(s) for s in stts]
    lifts = []
    for i, v in enumerate(vtts):
        read = visemes.to_visemes(v)
        acc = lambda said: 1 - experiment._edits(read, said) / len(said) if said else 0.0  # noqa: E731
        others = [j for j in range(len(stts)) if j != i]
        if nearest is not None:
            others = sorted(others, key=lambda j: abs(len(saids[j]) - len(saids[i])))[:nearest]
        lifts.append(float(acc(saids[i]) - np.mean([acc(saids[j]) for j in others])))
    return lifts


def viseme_lift(vtts: list[str], stts: list[str], nearest: int | None = None) -> float:
    return float(np.mean(viseme_lift_turns(vtts, stts, nearest)))


def _sentence_ranks(vtts: list[str], stts: list[str], pool: list[str] | None) -> list[tuple[float, float]]:
    candidates = list(stts) if pool is None else list(pool)
    if len(candidates) < 2:
        raise ValueError("sentence_id needs at least 2 candidate STTs")
    cand = [visemes.to_visemes(c) for c in candidates]
    rows = []
    for i, (v, s) in enumerate(zip(vtts, stts)):
        read = visemes.to_visemes(v)
        dist = lambda c: experiment._edits(read, c) / max(len(read), len(c), 1)  # noqa: E731
        own = dist(visemes.to_visemes(s))
        others = [dist(c) for j, c in enumerate(cand) if not (pool is None and j == i)]
        nearer = sum(d < own for d in others)
        ties = sum(d == own for d in others)
        top1 = 1 / (ties + 1) if nearer == 0 else 0.0
        rows.append((top1, (sum(d > own for d in others) + 0.5 * ties) / len(others)))
    return rows


def sentence_id_turns(vtts: list[str], stts: list[str], pool: list[str] | None = None) -> list[float]:
    """Per turn: the share of impostor STTs the turn's own STT beats in viseme distance to the read (ties half)."""
    return [p for _, p in _sentence_ranks(vtts, stts, pool)]


def sentence_id(vtts: list[str], stts: list[str], pool: list[str] | None = None) -> dict:
    """Closed-set identification: {"top1", "percentile", "n"}; `pool` (default the turns' own STTs) are the
    candidates; with a `pool`, each turn's own STT is added as the true one and the pool's are impostors."""
    rows = _sentence_ranks(vtts, stts, pool)
    n = len(stts) if pool is None else len(pool) + 1
    return {"top1": float(np.mean([t for t, _ in rows])), "percentile": float(np.mean([p for _, p in rows])), "n": n}


# ---------------------------------------------------------------- power


def paired_power(effect: float, sd: float, n: int, alpha: float = 0.05) -> float:
    """Power of a two-sided paired t test on n turns, when per-turn differences have mean `effect` and SD `sd`."""
    df, nc = n - 1, effect / sd * np.sqrt(n)
    critical = stats.t.ppf(1 - alpha / 2, df)
    power = stats.nct.sf(critical, df, nc) + stats.nct.cdf(-critical, df, nc)
    return 1.0 if np.isnan(power) else float(power)  # scipy's nct gives NaN only far out in the tail, where power -> 1


def turns_needed(effect: float, sd: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Smallest n (paired turns) with paired_power >= power."""
    n = 2
    while paired_power(effect, sd, n, alpha) < power:
        n += 1
    return n


def mde_paired(sd: float, n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable mean difference on n paired turns (the effect with exactly `power`)."""
    df = n - 1
    high = 2 * (stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)) * sd / np.sqrt(n)
    return float(optimize.brentq(lambda e: paired_power(e, sd, n, alpha) - power, 1e-12 * sd, high))


def sign_power(m: int, p_up: float, alpha: float = 0.05) -> float:
    """Power of the exact two-sided sign test on m non-tied turns, each up with probability p_up."""
    def p_value(k: int) -> float:
        low = min(k, m - k)
        return min(1.0, 2 * sum(comb(m, i) for i in range(low + 1)) / 2**m)
    return float(sum(comb(m, k) * p_up**k * (1 - p_up) ** (m - k) for k in range(m + 1) if p_value(k) <= alpha))


def sign_turns_needed(p_up: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Smallest number of non-tied turns with sign_power >= power. Divide by (1 - tie share) for total turns."""
    m = 1
    while sign_power(m, p_up, alpha) < power:
        m += 1
    return m


# ---------------------------------------------------------------- sample-to-sample noise


def _mode(texts: list[str]) -> str:
    return Counter(texts).most_common(1)[0][0] if texts else ""


def split_half_sd(samples: list[list[str]], stts: list[str], metric, pick=_mode) -> float:
    """Null SD of a per-turn difference: sqrt of the mean over turns of the mean squared metric difference between
    the picks of the two halves, over every floor(K/2) | rest split of the turn's K sample texts. Turns with K < 2
    are skipped. `metric(read, stt)` is a per-turn metric (per_turn_word_acc, per_turn_viseme_acc, ...)."""
    squares = []
    for texts, stt in zip(samples, stts):
        k = len(texts)
        if k < 2:
            continue
        diffs = []
        for half in combinations(range(k), k // 2):
            rest = [texts[i] for i in range(k) if i not in half]
            diffs.append(metric(pick([texts[i] for i in half]), stt) - metric(pick(rest), stt))
        squares.append(np.mean(np.square(diffs)))
    return float(np.sqrt(np.mean(squares)))
