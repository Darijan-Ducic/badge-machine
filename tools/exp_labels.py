"""Label quality: our STT text (a session's or a turn's) aligned into the official transcript of the address it came
from, and the metrics read off that alignment.

  words(text)        normalized word list, the same on both sides: lowercase, punctuation dropped, numbers spelled
                     out ("12" -> "twelve", the White House style); doubled words are KEPT (they are what we measure)
  align(hyp, ref)    semi-global word alignment: all of hyp against the best-matching span ref[start:end]
                     -> Alignment(start, end, edits, ops); ops has one entry per hyp word: "=" match, "s" substituted,
                     "i" inserted; deletions (ref words inside the span that hyp lacks) are counted in `deleted`
  doubles(hyp, a)    immediate repeats in hyp ("police police") that the reference does not have at that place
  boundary(a, b)     two consecutive turns' alignments -> {"lost": ref words in neither turn, "overlap": ref words
                     heard by both}
  press_text(html)   the address paragraphs of an obamawhitehouse.archives.gov press-office page (untrusted input,
                     parsed as text only)
"""

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _spell(n: int) -> list[str]:
    if n < 20:
        return [_ONES[n]]
    if n < 100:
        return [_TENS[n // 10]] + ([_ONES[n % 10]] if n % 10 else [])
    if n < 1000:
        return [_ONES[n // 100], "hundred"] + (_spell(n % 100) if n % 100 else [])
    for size, name in ((10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")):
        if n >= size:
            return _spell(n // size) + [name] + (_spell(n % size) if n % size else [])
    raise AssertionError(n)


def words(text: str) -> list[str]:
    text = text.lower().replace("’", "'").replace("‘", "'").replace("%", " percent ").replace("&", " and ")
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    out = []
    for token in re.findall(r"[a-z0-9']+", text):
        token = token.strip("'")
        if not token:
            continue
        out.extend(_spell(int(token)) if token.isdigit() and len(token) < 12 else [token])
    return out


@dataclass
class Alignment:
    start: int
    end: int
    edits: int
    ops: list[str] = field(default_factory=list)
    deleted: int = 0

    @property
    def ref_words(self) -> int:
        return self.end - self.start


def align(hyp: list[str], ref: list[str]) -> Alignment:
    m, n = len(hyp), len(ref)
    if m == 0:
        return Alignment(0, 0, 0)
    # d[i][j]: fewest edits aligning hyp[:i] to some ref[k:j]; a free start anywhere in ref (d[0][j] = 0).
    d = [[0] * (n + 1)] + [[i] + [0] * n for i in range(1, m + 1)]
    for i in range(1, m + 1):
        row, prev, h = d[i], d[i - 1], hyp[i - 1]
        for j in range(1, n + 1):
            row[j] = min(prev[j - 1] + (h != ref[j - 1]), prev[j] + 1, row[j - 1] + 1)
    end = min(range(n + 1), key=lambda j: (d[m][j], j))  # free end; the earliest on a tie
    i, j, ops, deleted = m, end, [], 0
    while i > 0:
        if j > 0 and d[i][j] == d[i - 1][j - 1] + (hyp[i - 1] != ref[j - 1]):
            ops.append("=" if hyp[i - 1] == ref[j - 1] else "s")
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j] + 1:
            ops.append("i")
            i -= 1
        else:
            deleted += 1
            j -= 1
    return Alignment(j, end, d[m][end], ops[::-1], deleted)


def doubles(hyp: list[str], alignment: Alignment) -> int:
    ops = alignment.ops
    return sum(hyp[k] == hyp[k - 1] and "i" in (ops[k - 1], ops[k]) for k in range(1, len(hyp)))


def boundary(first: Alignment, second: Alignment) -> dict:
    return {"lost": max(0, second.start - first.end), "overlap": max(0, first.end - second.start)}


class _Press(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth, self.paragraph, self.paragraphs = 0, None, []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "div" and (self.depth or "field-name-field-transcript" in (attrs.get("class") or "")):
            self.depth += 1
        elif tag == "p" and self.depth and "rtecenter" not in (attrs.get("class") or ""):
            self.paragraph = []

    def handle_endtag(self, tag):
        if tag == "div" and self.depth:
            self.depth -= 1
        elif tag == "p" and self.paragraph is not None:
            text = " ".join("".join(self.paragraph).split())
            if text and text != "###":
                self.paragraphs.append(text)
            self.paragraph = None

    def handle_data(self, data):
        if self.paragraph is not None:
            self.paragraph.append(data)


def press_text(html: str) -> str:
    parser = _Press()
    parser.feed(html)
    return " ".join(parser.paragraphs)


def repeats(hyp: list[str], alignment: Alignment, longest: int = 4) -> int:
    """Words of spurious immediate phrase repeats ("road to road to" -> 2): an n-gram (n <= longest) said again right
    after itself where the alignment marks a word of either copy as inserted, i.e. the reference has it once."""
    ops, k, count = alignment.ops, 1, 0
    while k < len(hyp):
        for n in range(min(longest, k, len(hyp) - k), 0, -1):
            if hyp[k:k + n] == hyp[k - n:k] and "i" in ops[k - n:k + n]:
                count, k = count + n, k + n
                break
        else:
            k += 1
    return count


def collapse_repeats(ws: list[str], longest: int = 4) -> list[str]:
    """Drop the second copy of every immediately repeated n-gram (n <= longest, longest first): "road to road to" ->
    "road to". The candidate label fix; it also drops a repeat that was really spoken."""
    out = []
    for w in ws:
        out.append(w)
        for n in range(min(longest, len(out) // 2), 0, -1):
            if out[-n:] == out[-2 * n:-n]:
                del out[-n:]
                break
    return out
