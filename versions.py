"""Version registry: every point on the accuracy chart is a system version (docs/self-improvement.md).

A version is a config over immutable raw outputs, run by some code. The baseline stores its full config; every other
version stores only its parent and ONE change: a typed config change (below), or a code change (kind "code": a git
commit in `code_commit`, config unchanged). `Registry.config` rebuilds a config by walking parents and re-applying
the config changes, then checks the rebuilt system against the stored hash. The hash (`version_hash`, stored as
`config_hash`) = sha256 over the canonical JSON of the full resolved config (`resolve`: defaults normalized, lists
ordered, numbers as floats), first 16 hex chars; when a code version is on the line, the nearest one's commit is
hashed with the config. On a config version `code_commit` is only provenance (the commit of the creating process).

Statuses: `candidate` (created, not yet evaluated), `evaluated` (evidence attached, unjudged), then the session's
judgment: `adopted`, `rejected`, or `rolled_back` (adopted once, then current moved back past it). Nothing is deleted.
Pointers: `current` = the best version so far, moved only by `adopt` and `rollback`; `live` = what users run, moved
only by `promote`, which the user approves. Every pointer move and every judgment is appended to the audit log
(who, when, why), served with the versions by `payload()`. Scripts never call these; the session does, through
tools/versions.py.

Storage: MongoDB (collection `versions`, one document per version with `_id` = id, plus `version_pointers` and the
append-only `version_audit`) when MONGODB_URI is set; JSON files under data/versions/ otherwise (<id>.json +
pointers.json + audit.jsonl). `open_registry("real")` is that default store; any other value is a directory.

Change space — the only changes the loop may make; anything else raises InvalidChange before it is applied:
  kind     op                  fields                                        config key
  tools    set_viseme_cost     pair "X|Y" (visemes-v1 codes, not REST), cost in [0, 1]    costs.sub (symmetric)
  tools    set_feature         feature (candidates.FEATURES), on (bool)      drop
  tools    set_pool_cap        pool_cap in 2..24 (∅ included)                pool_cap
  routing  set_sources         sources: non-empty subset of open/closed/gemini   sources
  routing  set_k               k in 1..5 (first K of the stored open samples)    k
  routing  set_gemini_gate     below in (0, 1], or null for always           gemini_below
  routing  set_stages          temperature < core < full (labeled clips, <= 1000; defaults stays 0)   stages
  context  curate_bank         add, drop: phrase lists, <= 10 each           bank
  context  set_prior_strength  prior_strength in [0, 4] (scales the prior feature)   prior_strength
  skills   set_prompt_hint     hint (<= 300 chars; "" clears) — PAID: re-calls Muse   prompt_hint
A change that leaves the config unchanged is refused too.
"""

import copy
import functools
import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import pymongo

import scorer
from candidates import FEATURES, POOL_CAP
from preheat import _normalize
from visemes import ALPHABET, ALPHABET_VERSION, load_costs

DATA = Path("data/versions")
KINDS = ("skills", "tools", "routing", "context")
PAID_KINDS = ("skills",)
SPLITS = ("train", "heldout", "test")
SOURCES = ("open", "closed", "gemini")
STAGE_NAMES = ("defaults", "temperature", "core", "full")
CODES = [c for c in ALPHABET if c != "REST"]  # REST is observed-only; to_visemes never produces it
MAX_K, MAX_STAGE, MAX_BANK_EDIT, MAX_BANK, MAX_PHRASE, MAX_HINT, MAX_PRIOR = 5, 1000, 10, 500, 80, 300, 4.0
CONFIG_KEYS = ("costs", "drop", "pool_cap", "sources", "k", "gemini_below", "stages", "bank", "prior_strength",
               "prompt_hint")
OPS = {  # op -> (kind, fields)
    "set_viseme_cost": ("tools", ("pair", "cost")),
    "set_feature": ("tools", ("feature", "on")),
    "set_pool_cap": ("tools", ("pool_cap",)),
    "set_sources": ("routing", ("sources",)),
    "set_k": ("routing", ("k",)),
    "set_gemini_gate": ("routing", ("below",)),
    "set_stages": ("routing", ("temperature", "core", "full")),
    "curate_bank": ("context", ("add", "drop")),
    "set_prior_strength": ("context", ("prior_strength",)),
    "set_prompt_hint": ("skills", ("hint",)),
}
PAYLOAD_KEYS = ("id", "parent", "t", "status", "kind", "change", "hypothesis", "author", "config_hash", "code_commit",
                "failures_cited", "metrics", "delta", "cost_usd", "reason", "pilot")
UNJUDGED = ("candidate", "evaluated")
COMMIT = re.compile(r"[0-9a-f]{40}")
REAL = "real"


class InvalidChange(ValueError):
    """A change (or config) outside the typed change space."""


# ---------------------------------------------------------------- typed values


def _int(value, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise InvalidChange(f"{name} = {value!r} must be an integer in [{low}, {high}]")
    return value


def _num(value, name: str, low: float, high: float, *, open_low: bool = False) -> float:
    ok = not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)
    if not ok or value > high or value < low or (open_low and value == low):
        raise InvalidChange(f"{name} = {value!r} must be a number in {'(' if open_low else '['}{low}, {high}]")
    return float(value)


def _pair(pair) -> str:
    x, sep, y = pair.partition("|") if isinstance(pair, str) else ("", "", "")
    if not sep or x not in CODES or y not in CODES or x == y:
        raise InvalidChange(f"pair {pair!r} must be 'X|Y' with two different codes from {CODES}")
    return "|".join(sorted((x, y), key=CODES.index))  # costs are symmetric: one key per unordered pair


def _phrases(values, name: str, most: int) -> list[str]:
    if not isinstance(values, list) or len(values) > most:
        raise InvalidChange(f"{name} must be a list of at most {most} phrases")
    kept: dict[str, str] = {}
    for text in values:
        if not isinstance(text, str) or not _normalize(text) or len(text) > MAX_PHRASE:
            raise InvalidChange(f"{name}: {text!r} must be a non-empty phrase of at most {MAX_PHRASE} characters")
        kept.setdefault(_normalize(text), " ".join(text.split()))
    return sorted(kept.values())


def _sources(values) -> list[str]:
    if not isinstance(values, list) or not values or not set(values) <= set(SOURCES) or len(set(values)) != len(values):
        raise InvalidChange(f"sources {values!r} must be a non-empty list of distinct names from {SOURCES}")
    return [s for s in SOURCES if s in values]


def _stages(temperature, core, full) -> dict:
    t, c, f = (_int(v, n, 1, MAX_STAGE) for v, n in ((temperature, "temperature"), (core, "core"), (full, "full")))
    if not t < c < f:
        raise InvalidChange(f"stages need 0 < temperature < core < full, got {t}, {c}, {f}")
    return {"defaults": 0, "temperature": t, "core": c, "full": f}


def _hint(value) -> str:
    if not isinstance(value, str) or len(value) > MAX_HINT:
        raise InvalidChange(f"hint must be a string of at most {MAX_HINT} characters")
    return value.strip()


# ---------------------------------------------------------------- configs


def resolve(config: dict) -> dict:
    """The validated, canonical form of a full config: equal configs resolve (and hash) equal."""
    if not isinstance(config, dict) or set(config) != set(CONFIG_KEYS):
        raise InvalidChange(f"a config has exactly the keys {CONFIG_KEYS}")
    costs = config["costs"]
    if not isinstance(costs, dict) or costs.get("alphabet") != ALPHABET_VERSION or not isinstance(costs.get("sub"), dict):
        raise InvalidChange(f"costs must be a {ALPHABET_VERSION} cost table with a 'sub' map")
    edits = {name: _num(costs.get(name), f"costs.{name}", 0, 1) for name in ("default_sub", "ins", "del")}
    sub: dict[str, float] = {}
    for pair, cost in costs["sub"].items():
        key, value = _pair(pair), _num(cost, f"costs.sub[{pair}]", 0, 1)
        if sub.get(key, value) != value:
            raise InvalidChange(f"costs.sub lists {pair!r} twice with different costs")
        sub[key] = value
    if not isinstance(config["drop"], list) or not set(config["drop"]) <= set(FEATURES):
        raise InvalidChange(f"drop must list names from {FEATURES}")
    stages = config["stages"]
    if not isinstance(stages, dict) or set(stages) != set(STAGE_NAMES) or stages["defaults"] != 0:
        raise InvalidChange(f"stages must give {STAGE_NAMES} with defaults at 0")
    bank = config["bank"]
    if not isinstance(bank, dict) or set(bank) != {"add", "drop"}:
        raise InvalidChange("bank must be {'add': [...], 'drop': [...]}")
    add, drop = _phrases(bank["add"], "bank.add", MAX_BANK), _phrases(bank["drop"], "bank.drop", MAX_BANK)
    if {_normalize(p) for p in add} & {_normalize(p) for p in drop}:
        raise InvalidChange("a phrase cannot be both added to and dropped from the bank")
    below = config["gemini_below"]
    return {
        "costs": {"alphabet": ALPHABET_VERSION, **edits,
                  "sub": {k: v for k, v in sorted(sub.items()) if v != edits["default_sub"]}},
        "drop": sorted(set(config["drop"])),
        "pool_cap": _int(config["pool_cap"], "pool_cap", 2, POOL_CAP),
        "sources": _sources(config["sources"]),
        "k": _int(config["k"], "k", 1, MAX_K),
        "gemini_below": None if below is None else _num(below, "gemini_below", 0, 1, open_low=True),
        "stages": _stages(stages["temperature"], stages["core"], stages["full"]),
        "bank": {"add": add, "drop": drop},
        "prior_strength": _num(config["prior_strength"], "prior_strength", 0, MAX_PRIOR),
        "prompt_hint": _hint(config["prompt_hint"]),
    }


def default_config() -> dict:
    """Today's system: data/viseme_costs.json, every feature, the full pool, all three calls, K=3, scorer stages."""
    return resolve({"costs": load_costs(), "drop": [], "pool_cap": POOL_CAP, "sources": list(SOURCES), "k": 3,
                    "gemini_below": None, "stages": {name: minimum for minimum, name in scorer.STAGES},
                    "bank": {"add": [], "drop": []}, "prior_strength": 1.0, "prompt_hint": ""})


def config_hash(config: dict) -> str:
    text = json.dumps(resolve(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def version_hash(config: dict, code: str | None = None) -> str:
    """The hash of a system: `config_hash` alone while no code version is on its line, else config + commit."""
    if code is None:
        return config_hash(config)
    text = json.dumps({"code": code, "config": resolve(config)}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- changes


def canonical(change) -> dict:
    """`change` with its fields type-checked and normalized (no config needed); InvalidChange if outside the space."""
    if not isinstance(change, dict) or change.get("op") not in OPS:
        op = change.get("op") if isinstance(change, dict) else change
        raise InvalidChange(f"unknown change {op!r}; the change space is {sorted(OPS)}")
    op = change["op"]
    fields = OPS[op][1]
    if set(change) != {"op", *fields}:
        raise InvalidChange(f"{op} takes exactly the fields {list(fields)}, got {sorted(set(change) - {'op'})}")
    if op == "set_viseme_cost":
        return {"op": op, "pair": _pair(change["pair"]), "cost": _num(change["cost"], "cost", 0, 1)}
    if op == "set_feature":
        if change["feature"] not in FEATURES or not isinstance(change["on"], bool):
            raise InvalidChange(f"set_feature needs a feature from {FEATURES} and on: true/false")
        return {"op": op, "feature": change["feature"], "on": change["on"]}
    if op == "set_pool_cap":
        return {"op": op, "pool_cap": _int(change["pool_cap"], "pool_cap", 2, POOL_CAP)}
    if op == "set_sources":
        return {"op": op, "sources": _sources(change["sources"])}
    if op == "set_k":
        return {"op": op, "k": _int(change["k"], "k", 1, MAX_K)}
    if op == "set_gemini_gate":
        below = change["below"]
        return {"op": op, "below": None if below is None else _num(below, "below", 0, 1, open_low=True)}
    if op == "set_stages":
        stages = _stages(change["temperature"], change["core"], change["full"])
        return {"op": op, **{k: stages[k] for k in ("temperature", "core", "full")}}
    if op == "curate_bank":
        add, drop = _phrases(change["add"], "add", MAX_BANK_EDIT), _phrases(change["drop"], "drop", MAX_BANK_EDIT)
        if not add and not drop:
            raise InvalidChange("curate_bank needs at least one phrase to add or drop")
        if {_normalize(p) for p in add} & {_normalize(p) for p in drop}:
            raise InvalidChange("curate_bank cannot add and drop the same phrase")
        return {"op": op, "add": add, "drop": drop}
    if op == "set_prior_strength":
        return {"op": op, "prior_strength": _num(change["prior_strength"], "prior_strength", 0, MAX_PRIOR)}
    return {"op": op, "hint": _hint(change["hint"])}


def _applied(config: dict, change: dict) -> dict:
    new = copy.deepcopy(config)
    op = change["op"]
    if op == "set_viseme_cost":
        new["costs"]["sub"] = {k: v for k, v in new["costs"]["sub"].items() if _pair(k) != change["pair"]}
        new["costs"]["sub"][change["pair"]] = change["cost"]
    elif op == "set_feature":
        drop = set(new["drop"])
        (drop.discard if change["on"] else drop.add)(change["feature"])
        new["drop"] = sorted(drop)
    elif op == "set_stages":
        new["stages"] = _stages(change["temperature"], change["core"], change["full"])
    elif op == "curate_bank":
        add = {_normalize(p): p for p in new["bank"]["add"]}
        drop = {_normalize(p): p for p in new["bank"]["drop"]}
        for text in change["add"]:
            drop.pop(_normalize(text), None)
            add[_normalize(text)] = text
        for text in change["drop"]:
            add.pop(_normalize(text), None)
            drop[_normalize(text)] = text
        new["bank"] = {"add": list(add.values()), "drop": list(drop.values())}
    else:
        key, field = {"set_pool_cap": ("pool_cap", "pool_cap"), "set_sources": ("sources", "sources"),
                      "set_k": ("k", "k"), "set_gemini_gate": ("gemini_below", "below"),
                      "set_prior_strength": ("prior_strength", "prior_strength"),
                      "set_prompt_hint": ("prompt_hint", "hint")}[op]
        new[key] = change[field]
    return new


def kind_of(change: dict) -> str:
    return OPS[canonical(change)["op"]][0]


def is_paid(change: dict) -> bool:
    return kind_of(change) in PAID_KINDS


def validate_change(change, config: dict, *, free_only: bool = False) -> dict:
    """The canonical form of `change` if it is in the change space and alters `config`; else InvalidChange."""
    typed = canonical(change)
    if free_only and OPS[typed["op"]][0] in PAID_KINDS:
        raise InvalidChange(f"{typed['op']} is a paid skills change (it re-calls Muse); this run is free-only")
    base = resolve(config)
    if resolve(_applied(base, typed)) == base:
        raise InvalidChange(f"{typed['op']} leaves the config unchanged")
    return typed


def apply(config: dict, change: dict) -> dict:
    typed = validate_change(change, config)
    return resolve(_applied(resolve(config), typed))


# ---------------------------------------------------------------- storage


class _Files:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _write(self, name: str, value: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / f".{name}.tmp"
        tmp.write_text(json.dumps(value, indent=1))
        os.replace(tmp, self.root / name)  # atomic: a crash never leaves half a record

    def all(self) -> list[dict]:
        if not self.root.exists():
            return []
        records = [json.loads(p.read_text()) for p in self.root.glob("*.json") if p.name != "pointers.json"]
        return sorted(records, key=lambda r: r["seq"])

    def get(self, vid: str) -> dict | None:
        path = self.root / f"{vid}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def put(self, record: dict) -> None:
        self._write(f"{record['id']}.json", record)

    def pointers(self) -> dict | None:
        path = self.root / "pointers.json"
        return json.loads(path.read_text()) if path.exists() else None

    def set_pointers(self, pointers: dict) -> None:
        self._write("pointers.json", pointers)

    def audit(self) -> list[dict]:
        path = self.root / "audit.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []

    def log(self, entry: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / "audit.jsonl", "a") as f:  # append-only: one line per move, never rewritten
            f.write(json.dumps(entry) + "\n")


class _Mongo:
    def __init__(self, db):
        self.versions = db["versions"]
        self.pointer_docs = db["version_pointers"]
        self.audit_docs = db["version_audit"]

    def all(self) -> list[dict]:
        return list(self.versions.find({}, {"_id": False}).sort("seq", pymongo.ASCENDING))

    def get(self, vid: str) -> dict | None:
        return self.versions.find_one({"_id": vid}, {"_id": False})

    def put(self, record: dict) -> None:
        self.versions.replace_one({"_id": record["id"]}, {"_id": record["id"], **record}, upsert=True)

    def pointers(self) -> dict | None:
        return self.pointer_docs.find_one({"_id": "pointers"}, {"_id": False})

    def set_pointers(self, pointers: dict) -> None:
        self.pointer_docs.replace_one({"_id": "pointers"}, {"_id": "pointers", **pointers}, upsert=True)

    def audit(self) -> list[dict]:
        return list(self.audit_docs.find({}, {"_id": False}).sort("seq", pymongo.ASCENDING))

    def log(self, entry: dict) -> None:
        self.audit_docs.insert_one(dict(entry))  # insert_one writes _id into the dict it gets


@functools.cache
def _mongo_client(uri: str):
    return pymongo.MongoClient(uri)


@functools.cache
def _code_commit() -> str:
    """The commit of the code this process loaded (read once: a long run keeps running the code it started with)."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _numeric(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


class Registry:
    """`store`: a directory (JSON files), a pymongo Database, or None for MONGODB_URI / data/versions."""

    def __init__(self, store=None):
        if store is None:
            uri = os.environ.get("MONGODB_URI")
            store = _mongo_client(uri)[os.environ.get("MONGODB_DB") or "badge_machine"] if uri else DATA
        self._store = _Files(store) if isinstance(store, str | Path) else _Mongo(store)

    # -------- reading

    def versions(self) -> list[dict]:
        return self._store.all()

    def get(self, vid: str) -> dict:
        record = self._store.get(vid)
        if record is None:
            raise KeyError(f"no version {vid!r}")
        return record

    def pointers(self) -> dict:
        found = self._store.pointers() or {}
        return {"live": found.get("live"), "current": found.get("current")}

    def code(self, vid: str) -> str | None:
        """The commit `vid` runs: its nearest code version's (itself included), or None if its line has none."""
        record = self.get(vid)
        while record["kind"] != "code":
            if record["parent"] is None:
                return None
            record = self.get(record["parent"])
        return record["code_commit"]

    def config(self, vid: str) -> dict:
        """The full config of `vid`: the baseline's config with every ancestor's config change applied (code
        versions change no config), checked against the stored hash of config + the nearest code commit."""
        chain, record, code = [], self.get(vid), None
        while record["parent"] is not None:
            if record["kind"] == "code":
                code = record["code_commit"] if code is None else code
            else:
                chain.append(record["change"])
            record = self.get(record["parent"])
        config = resolve(record["config"])
        for change in reversed(chain):
            config = apply(config, change)
        if version_hash(config, code) != self.get(vid)["config_hash"]:
            raise ValueError(f"{vid}: rebuilt config and code do not match its stored hash")
        return config

    def audit(self) -> list[dict]:
        """Every pointer move and judgment, oldest first: {seq, t, action, version, pointer, from, to, who, reason}."""
        return self._store.audit()

    def payload(self) -> dict:
        """The `GET /versions` body."""
        pointers = self.pointers()

        def status(record):
            if record["id"] == pointers["live"]:
                return "live"
            return "current" if record["id"] == pointers["current"] else record["status"]

        rows = [{**{k: record.get(k) for k in PAYLOAD_KEYS}, "status": status(record)} for record in self.versions()]
        return {"live": pointers["live"], "current": pointers["current"], "versions": rows, "audit": self.audit()}

    # -------- writing

    def _new(self, *, parent, kind, change, hypothesis, author, system_hash, failures_cited, cost_usd, status,
             evidence=None) -> dict:
        if not isinstance(author, str) or not author.strip():
            raise ValueError(f"author {author!r} must name who made the change")
        seq = len(self.versions()) + 1
        return {"id": f"sys-v{seq}", "seq": seq, "parent": parent, "t": time.time(), "status": status, "kind": kind,
                "change": change, "hypothesis": hypothesis, "author": author, "config_hash": system_hash,
                "code_commit": _code_commit(), "failures_cited": list(failures_cited),
                "metrics": {split: None for split in SPLITS}, "delta": {}, "cost_usd": float(cost_usd),
                "reason": None, "pilot": None, "evidence": evidence}

    def _log(self, action: str, vid: str, *, who: str, reason: str | None, pointer: str | None = None,
             to: str | None = None) -> None:
        entry = {"seq": len(self.audit()) + 1, "t": time.time(), "action": action, "version": vid,
                 "pointer": pointer, "from": None, "to": to, "who": who, "reason": reason}
        if pointer is not None:
            pointers = self.pointers()
            entry["from"] = pointers[pointer]
            self._store.set_pointers({**pointers, pointer: to})
        self._store.log(entry)

    def create_baseline(self, config: dict, *, hypothesis: str = "starting configuration",
                        author: str = "human") -> dict:
        if self.versions():
            raise ValueError("the registry already has a baseline")
        config = resolve(config)
        record = self._new(parent=None, kind="baseline", change={}, hypothesis=hypothesis, author=author,
                           system_hash=config_hash(config), failures_cited=(), cost_usd=0.0, status="adopted")
        record["config"] = config
        self._store.put(record)
        for pointer in ("live", "current"):
            self._log("baseline", record["id"], who=author, reason=hypothesis, pointer=pointer, to=record["id"])
        return record

    def create_candidate(self, parent: str, change: dict, *, hypothesis: str, author: str = "agent",
                         failures_cited=(), cost_usd: float = 0.0, evidence=None) -> dict:
        """A config change on `parent`: `candidate` until evaluated, or `evaluated` when created with evidence."""
        parent_config = self.config(parent)
        typed = validate_change(change, parent_config)
        record = self._new(parent=parent, kind=kind_of(typed), change=typed, hypothesis=hypothesis, author=author,
                           system_hash=version_hash(apply(parent_config, typed), self.code(parent)),
                           failures_cited=failures_cited, cost_usd=cost_usd, evidence=evidence,
                           status="candidate" if evidence is None else "evaluated")
        self._store.put(record)
        return record

    def create_code(self, parent: str, commit: str, *, hypothesis: str, evidence, author: str) -> dict:
        """A code change on `parent`: the system now runs `commit` (a full 40-hex sha), config unchanged."""
        if not isinstance(commit, str) or not COMMIT.fullmatch(commit):
            raise InvalidChange(f"code commit {commit!r} must be a full 40-character lowercase git sha")
        if commit == self.code(parent):
            raise InvalidChange(f"{parent} already runs {commit[:12]}; a code version must change the code")
        record = self._new(parent=parent, kind="code", change={}, hypothesis=hypothesis, author=author,
                           system_hash=version_hash(self.config(parent), commit), failures_cited=(), cost_usd=0.0,
                           evidence=evidence, status="evaluated")
        record["code_commit"] = commit
        self._store.put(record)
        return record

    def evaluated(self, vid: str, evidence) -> dict:
        """Attach the evaluation's evidence to a candidate and leave it unjudged (no pointer moves)."""
        if self.get(vid)["status"] != "candidate":
            raise ValueError(f"{vid} is not a candidate")
        return self._update(vid, status="evaluated", evidence=evidence)

    def _update(self, vid: str, **fields) -> dict:
        record = {**self.get(vid), **fields}
        self._store.put(record)
        return record

    def record_metrics(self, vid: str, split: str, metrics: dict) -> dict:
        """Store `metrics` for `split`; held-out metrics also set `delta` = this minus the parent's held-out."""
        if split not in SPLITS:
            raise ValueError(f"split {split!r} must be one of {SPLITS}")
        record = self.get(vid)
        fields = {"metrics": {**record["metrics"], split: dict(metrics)}}
        parent = self.get(record["parent"])["metrics"]["heldout"] if record["parent"] else None
        if split == "heldout" and parent:
            fields["delta"] = {k: metrics[k] - parent[k] for k in metrics
                               if k != "clips" and _numeric(metrics[k]) and _numeric(parent.get(k))}
        return self._update(vid, **fields)

    def add_cost(self, vid: str, usd: float) -> dict:
        return self._update(vid, cost_usd=self.get(vid)["cost_usd"] + float(usd))

    def set_pilot(self, vid: str, pilot: dict) -> dict:
        return self._update(vid, pilot=pilot)

    def adopt(self, vid: str, reason: str | None = None, *, who: str = "unknown") -> dict:
        """current := `vid`, which must be unjudged and built on the current version."""
        record = self.get(vid)
        current = self.pointers()["current"]
        if record["status"] not in UNJUDGED:
            raise ValueError(f"{vid} is {record['status']}; only an unjudged version can be adopted")
        if record["parent"] != current:
            raise ValueError(f"{vid} was measured against {record['parent']}, but current is now {current}")
        record = self._update(vid, status="adopted", reason=reason)
        self._log("adopt", vid, who=who, reason=reason, pointer="current", to=vid)
        return record

    def reject(self, vid: str, reason: str, *, who: str = "unknown") -> dict:
        """Judge `vid` not better; it keeps its evidence and moves no pointer. Undo an adoption with `rollback`."""
        if self.get(vid)["status"] not in UNJUDGED:
            raise ValueError(f"{vid} is {self.get(vid)['status']}; only an unjudged version can be rejected")
        record = self._update(vid, status="rejected", reason=reason)
        self._log("reject", vid, who=who, reason=reason)
        return record

    def rollback(self, to: str, reason: str, *, who: str = "unknown") -> dict:
        """current := `to`, an earlier version on current's line; every version between is marked rolled_back."""
        current = self.pointers()["current"]
        between, record = [], self.get(current)
        while record["id"] != to:
            if record["parent"] is None:
                raise ValueError(f"{to} is not an earlier version of current {current}")
            between.append(record["id"])
            record = self.get(record["parent"])
        if not between:
            raise ValueError(f"{to} is already current")
        for vid in between:
            self._update(vid, status="rolled_back")
        self._log("rollback", to, who=who, reason=reason, pointer="current", to=to)
        return record

    def promote(self, vid: str, reason: str | None = None, *, who: str = "unknown") -> dict:
        """Make `vid` live: the user's step. Only adopted versions (the baseline included) can go live."""
        record = self.get(vid)
        if record["status"] != "adopted":
            raise ValueError(f"{vid} is {record['status']}; only adopted versions can go live")
        self._log("promote", vid, who=who, reason=reason, pointer="live", to=vid)
        return record


def open_registry(where: str) -> Registry:
    """`real`: the registry the app serves (MongoDB when MONGODB_URI is set, else data/versions); else a directory."""
    return Registry(None if where == REAL else Path(where))
