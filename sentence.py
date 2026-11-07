"""Sentence post-processing for dictation turns: turn the K stored open samples into one sentence, with no model call.

The phrase path (candidates.featurize + scorer) picks one whole candidate from a pool, which suits short commands.
A 10-25-word dictation turn is never an exact candidate, so this module combines the samples word by word instead.
Every function works on word lists; `words(text)` normalizes with preheat._normalize first.

align(a, b)                 a minimum unit-cost edit alignment of two word lists: (x, y) pairs, None for a gap
mbr(hyps, weights, dist)    index of the hypothesis with the least weighted distance to all of them (ties: first)
rover(hyps, weights, gaps)  a confusion network built by aligning each hypothesis in turn; per column the word with
                            the most weight wins, the gap included as a competitor when gaps=True (ties: first seen)
slot_words(output)          one sample's best slot alternative per slot, in slot order
"""

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


def rover(hyps: list[list[str]], weights: list[float] | None = None, gaps: bool = True) -> list[str]:
    weights = [1.0] * len(hyps) if weights is None else weights
    columns: list[dict] = []  # word (None = gap) -> weight, in first-seen order
    before = 0.0  # total weight of the hypotheses already in the network
    for hyp, weight in zip(hyps, weights):
        def cost(c, j):
            if c is None:
                return 1  # a word no column has: a new column
            return int(None not in columns[c]) if j is None else int(hyp[j] not in columns[c])

        merged = []
        for c, j in _path(len(columns), len(hyp), cost):
            column = columns[c] if c is not None else ({None: before} if before else {})
            key = None if j is None else hyp[j]
            column[key] = column.get(key, 0.0) + weight
            merged.append(column)
        columns, before = merged, before + weight
    out = []
    for column in columns:
        options = [(key, w) for key, w in column.items() if gaps or key is not None]
        best = max(options, key=lambda item: item[1])[0]  # max keeps the first of equal weights
        if best is not None:
            out.append(best)
    return out


def slot_words(output: dict) -> list[str]:
    return [w for slot in output.get("slots") or [] if slot.get("alts")
            for w in words(max(slot["alts"], key=lambda a: a["s"])["w"])]
