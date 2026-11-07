"""Test helper: fake viseme functions and a simulator of raw per-clip model outputs.

The real `visemes.py` (CMUdict + `visemes-v1` codes) is built elsewhere; tests inject these FAKES instead:
every letter is its own "viseme" code and distance is a plain normalized Levenshtein. They keep the contract
(`to_visemes(text) -> list[str]`, `viseme_distance(a, b, costs=None) -> float` in [0, 1]) and are easy to do by hand.
"""

from preheat import _normalize


def fake_to_visemes(text: str) -> list[str]:
    """FAKE `visemes.to_visemes`: letters as codes, spaces and punctuation dropped, consecutive repeats collapsed."""
    letters = [ch for ch in _normalize(text) if ch.isalpha()]
    return [ch for i, ch in enumerate(letters) if i == 0 or ch != letters[i - 1]]


def fake_viseme_distance(a: list[str], b: list[str], costs=None) -> float:
    """FAKE `visemes.viseme_distance`: unit-cost Levenshtein divided by the longer length (0 for two empties)."""
    if not a and not b:
        return 0.0
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1] / max(len(a), len(b))
