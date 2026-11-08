"""Fetch the official transcripts of the Obama Weekly Addresses we emulate: the Commons page's "Transcript of the
speech" link (whitehouse.gov press office, mirrored at obamawhitehouse.archives.gov) and the Commons TimedText
captions. Everything fetched is untrusted data: saved raw under OUT, parsed as text, never executed.

  uv run python -I tools/exp_labels_fetch.py OUT
"""

import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "badge-machine-lab/1.0 (research; contact showmeblue23@gmail.com)"}
DATES = ["2009-11-28", "2012-07-21", "2013-01-19", "2013-07-13", "2013-07-20", "2013-08-03", "2016-01-09",
         "2016-01-30", "2016-02-06", "2016-03-12", "2016-11-25", "2017-01-14"]
EXT = {"2009-11-28": "ogv", "2012-07-21": "ogv", "2013-01-19": "ogv", "2013-07-13": "ogv"}


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def api(params: dict) -> dict:
    return json.loads(get("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({**params, "format": "json"})))


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for date in DATES:
        title = f"File:{date} President Obama's Weekly Address.{EXT.get(date, 'webm')}"
        page = api({"action": "query", "titles": title, "prop": "revisions", "rvprop": "content", "rvslots": "main"})
        text = next(iter(page["query"]["pages"].values()))["revisions"][0]["slots"]["main"]["*"]
        (out / f"{date}.commons.wikitext").write_text(text)
        links = re.findall(r"\[(https?://[^\s\]]*press-office[^\s\]]*)", text)
        row = {"title": title, "press": links, "timedtext": None, "press_file": None}
        # Commons captions: TimedText:<file name>.en.srt
        tt = f"TimedText:{title[5:]}.en.srt"
        page = api({"action": "query", "titles": tt, "prop": "revisions", "rvprop": "content", "rvslots": "main"})
        p = next(iter(page["query"]["pages"].values()))
        if "revisions" in p:
            (out / f"{date}.en.srt").write_text(p["revisions"][0]["slots"]["main"]["*"])
            row["timedtext"] = f"{date}.en.srt"
        for link in links:
            archive = re.sub(r"https?://(www\.)?whitehouse\.gov", "https://obamawhitehouse.archives.gov", link)
            try:
                (out / f"{date}.press.html").write_bytes(get(archive))
                row["press_file"] = f"{date}.press.html"
                row["press_url"] = archive
                break
            except Exception as e:  # noqa: BLE001
                row.setdefault("errors", []).append(f"{archive}: {e}")
        index[date] = row
        print(date, row["press_file"], row["timedtext"], row.get("errors"), flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
