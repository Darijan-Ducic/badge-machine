"""Sentence post-processing for dictation turns: turn the K stored open samples into one sentence, with no model call.

The phrase path (candidates.featurize + scorer) picks one whole candidate from a pool, which suits short commands.
A 10-25-word dictation turn is never an exact candidate, so this module combines the samples word by word instead.
Every function works on word lists; `words(text)` normalizes with preheat._normalize first.

align(a, b)                 a minimum unit-cost edit alignment of two word lists: (x, y) pairs, None for a gap
mbr(hyps, weights, dist)    index of the hypothesis with the least weighted distance to all of them (ties: first)
network(hyps, weights)      the confusion network: each hypothesis aligned in turn, columns {word or None: weight}
rover(hyps, weights, gaps)  per network column the word with the most weight wins, the gap included as a competitor
                            when gaps=True (ties: first seen)
slot_words(output)          one sample's best slot alternative per slot, in slot order
"""

from statistics import median

from preheat import _normalize


def words(text: str | None) -> list[str]:
    return _normalize(text or "").split()


def edits(a: list, b: list) -> int:
    """Unit-cost Levenshtein distance (the word-accuracy metric's)."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def _path(n: int, m: int, cost) -> list[tuple[int | None, int | None]]:
    """Cheapest monotone path from (0, 0) to (n, m) in index pairs (i, j), None for a gap on that side.

    cost(i, j) for the diagonal step, cost(i, None) for consuming row item i alone, cost(None, j) for column item j.
    Ties prefer the diagonal, then the row-only step."""
    total = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        total[i][0] = total[i - 1][0] + cost(i - 1, None)
    for j in range(1, m + 1):
        total[0][j] = total[0][j - 1] + cost(None, j - 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            total[i][j] = min(total[i - 1][j - 1] + cost(i - 1, j - 1), total[i - 1][j] + cost(i - 1, None),
                              total[i][j - 1] + cost(None, j - 1))
    steps, i, j = [], n, m
    while i or j:
        if i and j and total[i][j] == total[i - 1][j - 1] + cost(i - 1, j - 1):
            i, j = i - 1, j - 1
            steps.append((i, j))
        elif i and total[i][j] == total[i - 1][j] + cost(i - 1, None):
            i -= 1
            steps.append((i, None))
        else:
            j -= 1
            steps.append((None, j))
    return steps[::-1]


def align(a: list[str], b: list[str]) -> list[tuple[str | None, str | None]]:
    def cost(i, j):
        return 1 if i is None or j is None else int(a[i] != b[j])

    return [(None if i is None else a[i], None if j is None else b[j]) for i, j in _path(len(a), len(b), cost)]


def mbr(hyps: list[list[str]], weights: list[float] | None = None, distance=None) -> int:
    weights = [1.0] * len(hyps) if weights is None else weights
    distance = distance or edits
    risks = [sum(w * distance(h, other) for other, w in zip(hyps, weights)) for h in hyps]
    return min(range(len(hyps)), key=lambda i: (risks[i], i))


def network(hyps: list[list[str]], weights: list[float] | None = None) -> list[dict]:
    """The confusion network: columns {word (None = gap): weight}, in first-seen order, aligning each hypothesis in
    turn to the network so far (a column already holding a gap is skipped for free; a word no column has is a new
    column, where every earlier hypothesis holds a gap)."""
    weights = [1.0] * len(hyps) if weights is None else weights
    columns: list[dict] = []
    before = 0.0  # total weight of the hypotheses already in the network
    for hyp, weight in zip(hyps, weights):
        def cost(c, j, hyp=hyp, columns=columns):
            if c is None:
                return 1
            return int(None not in columns[c]) if j is None else int(hyp[j] not in columns[c])

        merged = []
        for c, j in _path(len(columns), len(hyp), cost):
            column = columns[c] if c is not None else ({None: before} if before else {})
            key = None if j is None else hyp[j]
            column[key] = column.get(key, 0.0) + weight
            merged.append(column)
        columns, before = merged, before + weight
    return columns


def rover(hyps: list[list[str]], weights: list[float] | None = None, gaps: bool = True) -> list[str]:
    out = []
    for column in network(hyps, weights):
        options = [(key, w) for key, w in column.items() if gaps or key is not None]
        best = max(options, key=lambda item: item[1])[0]  # max keeps the first of equal weights
        if best is not None:
            out.append(best)
    return out


def _top_words(output: dict) -> tuple[list[str], float] | None:
    hyps = [(words(h["text"]), h["s"]) for h in output.get("hypotheses") or []]
    hyps = [h for h in hyps if h[0]]
    return max(hyps, key=lambda h: h[1]) if hyps else None  # max keeps the first of equal scores


def read(outputs: list[dict], phrases: list[str]) -> tuple[str, float] | None:
    """The sentence path for one clip's valid open outputs, or None when the clip belongs on the phrase path.

    A clip is a sentence turn when the median word count of its samples' top hypotheses exceeds the longest phrase in
    the phrase list. Its sentence is the ROVER (gaps on) of those top hypotheses, highest score first; its confidence
    is the mean share of its column's weight each chosen word holds."""
    tops = [t for t in map(_top_words, outputs) if t is not None]
    longest = max((len(words(p)) for p in phrases), default=0)
    if not tops or median(len(t[0]) for t in tops) <= longest:
        return None
    tops.sort(key=lambda t: -t[1])  # stable: equal scores keep sample order
    shares = []
    for column in network([t[0] for t in tops]):
        word, weight = max(column.items(), key=lambda item: item[1])
        if word is not None:
            shares.append((word, weight / sum(column.values())))
    if not shares:
        return None
    return " ".join(w for w, _ in shares), sum(s for _, s in shares) / len(shares)


def slot_words(output: dict) -> list[str]:
    return [w for slot in output.get("slots") or [] if slot.get("alts")
            for w in words(max(slot["alts"], key=lambda a: a["s"])["w"])]
