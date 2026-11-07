"""Viseme alphabet visemes-v1: text -> viseme codes, and a weighted edit distance between code sequences.

Spec: docs/uncertainty-design.md ("Viseme alphabet visemes-v1", "Module contracts"). Changing the alphabet or
the phoneme mapping makes old data incomparable — bump ALPHABET_VERSION if you do.

to_visemes(text)
    1. Normalize: typographic apostrophes (’ ‘) become ', lowercase, every character that is not a letter, digit
       or apostrophe becomes a word break (so "well-known" is two words), apostrophes are kept only inside a
       word ("don't", "we're"; quotes around a word are dropped).
    2. Each word: first CMUdict pronunciation (cmudict package, bundled data, no network), stress digits
       stripped, each ARPAbet phoneme mapped to its code. REST is never produced; it is observed-only.
    3. Out-of-vocabulary words (names, slang) use a deterministic letter-to-viseme fallback (_LETTERS):
       digraphs first (th sh ch ph ng wh ck qu oo), then single letters; digits and other characters are
       skipped. It is APPROXIMATE — English spelling is not phonetic (silent letters, soft c/g, vowel quality).
    4. Consecutive duplicate codes are collapsed across the whole phrase, including across word boundaries.

viseme_distance(a, b, costs=None)
    Weighted Levenshtein turning sequence a into sequence b, divided by max(len(a), len(b)); 0 for two empty
    sequences. With every cost in [0, 1] the result is in [0, 1]. costs=None uses load_costs() (read once).

Costs file data/viseme_costs.json (offline agents edit this):
    {
      "alphabet": "visemes-v1",   # must equal ALPHABET_VERSION
      "default_sub": 1,           # substitution cost for any pair of different codes not listed in "sub"
      "ins": 1,                   # cost of a code present in b but not in a
      "del": 1,                   # cost of a code present in a but not in b
      "sub": {"P|F": 0.6}         # per-pair substitution cost, key "<code in a>|<code in b>"
    }
    A "X|Y" entry also covers Y -> X unless "Y|X" is listed too (then each direction uses its own entry).
    Substituting a code for itself always costs 0. All costs must be in [0, 1]; codes must be in ALPHABET.
    Keys starting with "_" are ignored (free-form notes).
"""

import json
from functools import cache
from pathlib import Path

import cmudict

ALPHABET_VERSION = "visemes-v1"
ALPHABET: list[str] = ["P", "F", "TH", "W", "R", "CH", "S", "T", "K", "A", "E", "O", "REST"]

# ARPAbet column of the visemes-v1 table — all 39 CMUdict phonemes.
_ARPABET = {
    "P": "P", "B": "P", "M": "P",
    "F": "F", "V": "F",
    "TH": "TH", "DH": "TH",
    "W": "W", "UW": "W", "UH": "W",
    "R": "R", "ER": "R",
    "CH": "CH", "JH": "CH", "SH": "CH", "ZH": "CH",
    "S": "S", "Z": "S",
    "T": "T", "D": "T", "N": "T", "L": "T",
    "K": "K", "G": "K", "HH": "K", "NG": "K",
    "AA": "A", "AE": "A", "AH": "A", "AY": "A", "AW": "A",
    "EH": "E", "EY": "E", "IH": "E", "IY": "E", "Y": "E",
    "AO": "O", "OW": "O", "OY": "O",
}  # fmt: skip

# Out-of-vocabulary fallback (approximate). Two-letter keys are tried before single letters.
_LETTERS = {
    "th": ["TH"], "sh": ["CH"], "ch": ["CH"], "ph": ["F"], "ng": ["K"], "wh": ["W"], "ck": ["K"],
    "qu": ["K", "W"], "oo": ["W"],
    "a": ["A"], "b": ["P"], "c": ["K"], "d": ["T"], "e": ["E"], "f": ["F"], "g": ["K"], "h": ["K"],
    "i": ["E"], "j": ["CH"], "k": ["K"], "l": ["T"], "m": ["P"], "n": ["T"], "o": ["O"], "p": ["P"],
    "q": ["K"], "r": ["R"], "s": ["S"], "t": ["T"], "u": ["A"], "v": ["F"], "w": ["W"], "x": ["K", "S"],
    "y": ["E"], "z": ["S"],
}  # fmt: skip


@cache
def _lexicon() -> dict[str, list[list[str]]]:
    return cmudict.dict()


def _words(text: str) -> list[str]:
    text = text.replace("’", "'").replace("‘", "'").lower()
    cleaned = "".join(c if c.isalnum() or c == "'" else " " for c in text)
    return [w for w in (token.strip("'") for token in cleaned.split()) if w]


def _fallback(word: str) -> list[str]:
    codes, i = [], 0
    while i < len(word):
        if word[i : i + 2] in _LETTERS:
            codes += _LETTERS[word[i : i + 2]]
            i += 2
        else:
            codes += _LETTERS.get(word[i], [])
            i += 1
    return codes


def to_visemes(text: str) -> list[str]:
    codes: list[str] = []
    for word in _words(text):
        prons = _lexicon().get(word)
        if prons:
            word_codes = [_ARPABET[phone.rstrip("012")] for phone in prons[0]]
        else:
            word_codes = _fallback(word)
        for code in word_codes:
            if not codes or codes[-1] != code:
                codes.append(code)
    return codes


def load_costs(path: Path = Path("data/viseme_costs.json")) -> dict:
    costs = {k: v for k, v in json.loads(Path(path).read_text()).items() if not k.startswith("_")}
    if costs.get("alphabet") != ALPHABET_VERSION:
        raise ValueError(f"{path}: alphabet {costs.get('alphabet')!r} != {ALPHABET_VERSION!r}")
    values = {name: costs[name] for name in ("default_sub", "ins", "del")}
    for pair, value in costs["sub"].items():
        x, sep, y = pair.partition("|")
        if not sep or x not in ALPHABET or y not in ALPHABET or x == y:
            raise ValueError(f"{path}: bad pair {pair!r}; use '<code>|<other code>' with codes from ALPHABET")
        values[pair] = value
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 1:
            raise ValueError(f"{path}: cost {name!r} = {value!r} must be a number in [0, 1]")
    return costs


@cache
def _default_costs() -> dict:
    return load_costs()


def viseme_distance(a: list[str], b: list[str], costs: dict | None = None) -> float:
    if costs is None:
        costs = _default_costs()
    default_sub, ins, dele, sub = costs["default_sub"], costs["ins"], costs["del"], costs["sub"]

    def sub_cost(x: str, y: str) -> float:
        if x == y:
            return 0
        return sub.get(f"{x}|{y}", sub.get(f"{y}|{x}", default_sub))

    longest = max(len(a), len(b))
    if longest == 0:
        return 0.0
    prev = [j * ins for j in range(len(b) + 1)]
    for i in range(1, len(a) + 1):
        cur = [i * dele] + [0.0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + dele, cur[j - 1] + ins, prev[j - 1] + sub_cost(a[i - 1], b[j - 1]))
        prev = cur
    return prev[-1] / longest
