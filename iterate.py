"""The evaluation loop (docs/self-improvement.md): train failures -> ONE typed change -> held-out evidence, unjudged.

One iteration:
  (a) failures   evaluator.failures(train split, current config): clips where VTT != STT, with their candidate pools
  (b) propose    the proposer receives ONLY the context {failures summary (bounded, clustered), current config, the
                 current version's train metrics, tried-ideas memory, change space} — never a held-out or test clip,
                 id or text — and returns ONE change + hypothesis + the failure ids it cites. GivenChange is the
                 session's own change (tools/iterate.py --proposer none, the default); MuseProposer asks Muse Spark.
  (c) check      before anything is evaluated: in the change space (versions.validate_change), not a change already
                 rejected, not a system already evaluated, runnable by the evaluator, a paid pilot within budget
  (d) build      the candidate version in the registry, on the current version
  (e) evaluate   train (for the chart) and heldout; the evidence (both splits, the parent's, the deltas in points,
                 the records' count + sha256) goes on the version, which becomes `evaluated`
  (f) log        then repeat, until `iterations`, a dry proposer, or the budget
The loop never judges: it never adopts, rejects, rolls back or promotes, so `current` and `live` never move here. The
session reads the evidence and judges with tools/versions.py. Every proposal therefore builds on the same current.
Paid (skills) changes run a PILOT_CLIPS-clip pilot on held-out first (candidate vs parent, same clips) and get the
full evaluation only if the pilot improves word_acc and the estimated cost fits the budget (a spend guard, recorded as
a note in the evidence); free-only runs never offer them. Budget = this invocation's dollars: proposer calls + paid
evaluations (free replays cost nothing).
Resumable: all state is the registry. The baseline is created and evaluated once; a candidate a crash left
unevaluated is evaluated on restart instead of being proposed again. The test split is evaluated only by `final()`
(`tools/iterate.py --final`), once per version; the loop's evaluator refuses it.
"""

import copy
import hashlib
import json
import math
from collections import Counter

from candidates import FEATURES, NONE, POOL_CAP
from evaluator import PilotUnsupported, split
from muse_video import MODEL
from versions import (CODES, MAX_K, MAX_STAGE, OPS, PAID_KINDS, SOURCES, InvalidChange, apply, canonical,
                      config_hash, default_config, is_paid, kind_of, validate_change, version_hash)

PILOT_CLIPS = 20
MAX_EXAMPLES = 12
MAX_CONFUSIONS = 5
DELTA_KEYS = ("word_acc", "sentence_acc", "viseme_acc", "top3_acc")
MUSE_PRICE = (1.25, 4.25)  # $ per 1M input / output tokens, as experiment.PRICES["muse"]
CLUSTERS = {
    "pool_miss": "STT was not among the candidates at all: no source proposed it",
    "ranked_wrong": "STT was a candidate but another candidate scored higher",
    "abstained": "STT was a candidate but ∅ (nothing / none of these) won",
}
FIELD_DOCS = {
    "set_viseme_cost": f"pair: 'X|Y', two different codes from {' '.join(CODES)}; cost 0..1 (1 = unrelated shapes, "
                       "lower = look-alikes). Changes viseme distances: the viseme_sim feature and which bank phrases "
                       "are 'nearest' candidates",
    "set_feature": f"feature: one of {', '.join(FEATURES)}; on: true/false (off = the column is zeroed everywhere)",
    "set_pool_cap": f"pool_cap: 2..{POOL_CAP} candidates kept per clip, ∅ included (first ones in source order)",
    "set_sources": f"sources: non-empty subset of {list(SOURCES)} — which model calls feed the pool and features",
    "set_k": f"k: 1..{MAX_K} — how many of the stored Muse open samples are used (more = steadier votes, more cost)",
    "set_gemini_gate": "below: 0..1 or null — use Gemini only when the share of Muse samples agreeing on the top "
                       "hypothesis is below this; null = always",
    "set_stages": f"temperature < core < full: labeled-clip counts (1..{MAX_STAGE}) at which the scorer unlocks a "
                  "fitted temperature, then core features, then all features",
    "curate_bank": "add, drop: up to 10 phrases each — the user's phrase bank (bank-nearest candidates, in_bank, "
                   "prior). Add only phrases seen as STT in the failures summary",
    "set_prior_strength": "prior_strength: 0..4 — multiplies the phrase-history prior feature",
    "set_prompt_hint": "hint: up to 300 characters appended to the Muse open-call prompt (PAID: re-calls Muse on "
                       "every evaluated clip; runs a 20-clip pilot first)",
}


# ---------------------------------------------------------------- evidence, summary, memory


def _pts(fraction: float) -> float:
    return round(100 * fraction, 6)  # rounding keeps 0.51 - 0.50 from reading as 1.0000000000000009 points


def heldout_delta(parent: dict, candidate: dict) -> dict:
    """candidate - parent for each accuracy metric, in points. Evidence only: nothing here decides adoption."""
    return {k: _pts(candidate[k] - parent[k]) for k in DELTA_KEYS}


def records_digest(records: list[dict]) -> dict:
    """Count + sha256 of the records as JSON, the same bytes tools/compare.py hashes for a snapshot."""
    return {"count": len(records), "sha256": hashlib.sha256(json.dumps(records, default=str).encode()).hexdigest()}


def _cluster(failure: dict) -> str:
    if not failure["stt_in_pool"]:
        return "pool_miss"
    return "abstained" if failure["vtt"] == NONE else "ranked_wrong"


def summarize(failures: list[dict], n_clips: int, max_examples: int = MAX_EXAMPLES) -> dict:
    """A bounded, clustered failure summary: counts + top STT->VTT confusions per cluster, and a few examples
    (newest first, taken round-robin across clusters so the small clusters are seen too)."""
    groups = {name: [f for f in failures if _cluster(f) == name] for name in CLUSTERS}
    clusters = sorted(({"type": name, "meaning": CLUSTERS[name], "count": len(found),
                        "top_confusions": [{"stt": stt, "vtt": vtt, "count": n} for (stt, vtt), n in
                                           Counter((f["stt"], f["vtt"]) for f in found).most_common(MAX_CONFUSIONS)]}
                       for name, found in groups.items() if found), key=lambda c: -c["count"])
    queues = [list(reversed(groups[c["type"]])) for c in clusters]
    examples = []
    while len(examples) < max_examples and any(queues):
        for queue in queues:
            if queue and len(examples) < max_examples:
                f = queue.pop(0)
                examples.append({k: f[k] for k in ("id", "stt", "vtt", "stt_in_pool", "candidates")})
    return {"train_clips": n_clips, "failures": len(failures), "clusters": clusters, "examples": examples}


def tried_ideas(versions: list[dict], refused: list[dict]) -> list[dict]:
    """Every past change with its held-out delta (points) and why it was rejected — plus this run's refusals."""
    memory = []
    for v in versions[1:]:
        delta = v.get("delta") or {}
        memory.append({"id": v["id"], "kind": v["kind"], "change": v["change"], "hypothesis": v["hypothesis"],
                       "status": v["status"], "reason": v.get("reason"),
                       "delta": {k: round(100 * delta[k], 2) for k in ("word_acc", "viseme_acc") if k in delta}})
    return memory + [{"id": None, "kind": r["kind"], "change": r["change"], "hypothesis": r["hypothesis"],
                      "status": "refused", "reason": r["reason"], "delta": {}} for r in refused]


def change_space(free_only: bool) -> list[dict]:
    return [{"op": op, "kind": kind, "fields": FIELD_DOCS[op], "paid": kind in PAID_KINDS}
            for op, (kind, _) in OPS.items() if not (free_only and kind in PAID_KINDS)]


def _describe(change) -> str:
    if not isinstance(change, dict):
        return repr(change)
    args = " ".join(f"{k}={json.dumps(v)}" for k, v in change.items() if k != "op")
    return f"{change.get('op', '?')} {args}".strip()


# ---------------------------------------------------------------- proposers


class FakeProposer:
    """Deterministic proposer for tests and demos: walks a script of proposals (dicts, or callables of the context
    returning a dict or None). With respect_memory it skips any change already in the tried-ideas memory."""

    def __init__(self, script, respect_memory: bool = True):
        self.script = list(script)
        self.respect_memory = respect_memory
        self.calls, self.checks = [], []  # every context received (deep copies), and the check callables
        self._next = 0

    @staticmethod
    def _same(a, b) -> bool:
        try:
            return canonical(a) == canonical(b)
        except InvalidChange:
            return a == b

    def propose(self, context: dict, check=None) -> dict | None:
        self.calls.append(copy.deepcopy(context))
        self.checks.append(check)
        tried = [m["change"] for m in context["memory"]]
        while self._next < len(self.script):
            item = self.script[self._next]
            self._next += 1
            proposal = item(context) if callable(item) else copy.deepcopy(item)
            if proposal is None or (self.respect_memory and any(self._same(proposal["change"], t) for t in tried)):
                continue
            return {**proposal, "cost_usd": 0.0}
        return None


class GivenChange:
    """The session's own ONE change (tools/iterate.py --proposer none): proposed once, checked and evaluated like
    any other proposal, then nothing more."""

    def __init__(self, change, hypothesis: str = ""):
        self.proposal = {"change": change, "hypothesis": hypothesis, "failures_cited": []}
        self.given = False

    def propose(self, context: dict, check=None) -> dict | None:
        if self.given:
            return None
        self.given = True
        return {**copy.deepcopy(self.proposal), "cost_usd": 0.0}


def _obj(**properties) -> dict:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _field_schemas() -> dict:
    integer = {"type": "integer", "minimum": 1, "maximum": MAX_STAGE}
    pairs = [f"{x}|{y}" for i, x in enumerate(CODES) for y in CODES[i + 1:]]
    phrases = {"type": "array", "items": {"type": "string"}, "maxItems": 10}
    return {
        "set_viseme_cost": {"pair": {"type": "string", "enum": pairs}, "cost": {"type": "number", "minimum": 0,
                                                                                "maximum": 1}},
        "set_feature": {"feature": {"type": "string", "enum": FEATURES}, "on": {"type": "boolean"}},
        "set_pool_cap": {"pool_cap": {"type": "integer", "minimum": 2, "maximum": POOL_CAP}},
        "set_sources": {"sources": {"type": "array", "items": {"type": "string", "enum": list(SOURCES)},
                                    "minItems": 1, "maxItems": len(SOURCES)}},
        "set_k": {"k": {"type": "integer", "minimum": 1, "maximum": MAX_K}},
        "set_gemini_gate": {"below": {"anyOf": [{"type": "number", "minimum": 0, "maximum": 1}, {"type": "null"}]}},
        "set_stages": {"temperature": integer, "core": integer, "full": integer},
        "curate_bank": {"add": phrases, "drop": phrases},
        "set_prior_strength": {"prior_strength": {"type": "number", "minimum": 0, "maximum": 4}},
        "set_prompt_hint": {"hint": {"type": "string"}},
    }


def proposal_schema(context: dict) -> dict:
    """The proposal schema in reasoning order: observed pattern -> hypothesis -> the ONE change -> cited clips."""
    fields = _field_schemas()
    branches = [_obj(op={"type": "string", "enum": [s["op"]]}, **fields[s["op"]]) for s in context["change_space"]]
    shown = [e["id"] for e in context["failures"]["examples"]]
    cited = ({"type": "array", "items": {"type": "string", "enum": shown}} if shown
             else {"type": "array", "items": {"type": "string"}, "maxItems": 0})
    return _obj(pattern={"type": "string"}, hypothesis={"type": "string"}, change={"anyOf": branches},
                failures_cited=cited)


SYSTEM_PROMPT = """You are the iteration agent of a lip-reading system. VTT (video-to-text) is what the system reads \
from a muted mouth crop; STT (speech-to-text) is the reference label. You improve the system ONE typed change at a \
time. Every change you propose is evaluated on held-out clips you never see; a reviewer reads that evidence and \
adopts the change only if held-out word accuracy rises without viseme accuracy falling. A rejected idea stays \
rejected: proposing it again wastes an iteration."""

USER_PROMPT = """<pipeline>
Per clip: Muse Spark open samples (up to 5 stored; the config's k uses the first k) each give visemes, word-slot \
alternatives and whole-sentence hypotheses; a Muse closed call scores the phrase bank plus "none of these"; Gemini \
scores the phrase list. The candidate pool = hypotheses, slot recombinations, closed top 3, Gemini top 3, the 3 bank \
phrases nearest in visemes, and ∅. Each candidate gets 13 features ({features}); a staged conditional-logit scorer \
picks VTT. A failure is a training clip where VTT != STT.
</pipeline>

<inputs>
Failure summary of the TRAINING stream ({train_clips} clips, {n_failures} failures), clustered:
{failures}

Current configuration ({current_id}):
{config}

Current version on the training stream: {train_metrics}

Tried-ideas memory (every past change, its held-out delta in points, and why it was rejected or refused):
{memory}

Change space (the ONLY changes you may make):
{change_space}
</inputs>

<task>
Propose exactly ONE change from the change space that you expect to fix the largest group of failures you can \
explain, and that is not already in the memory as rejected or refused.
</task>

<output_spec>
JSON with, in this order:
- pattern: the failure cluster you target, quoting its type, count and 1-3 of its top confusions from the summary.
- hypothesis: the mechanism — why this knob fixes those clips (which feature, source or candidate it changes and \
how), and the expected effect on held-out word accuracy. 1-3 sentences.
- change: one op with its fields.
- failures_cited: 1-5 ids from the summary's examples that this change should fix.
</output_spec>

<the_bar>
Required (choose another change if your first idea fails any):
1. pattern names a cluster that exists in the summary, with its count.
2. hypothesis states a concrete mechanism tied to the cited clips, not "might help".
3. change is not identical to any memory entry with status rejected or refused, and does not undo an adopted change.
4. every phrase added to the bank appears as an STT text in the summary; never invent phrases or clip ids.
5. failures_cited lists only ids shown in the summary's examples.
Must NOT: bundle several ideas into one change; repeat a rejected idea with a cosmetically different value unless \
the hypothesis says why the new value behaves differently.
</the_bar>

<guidance>
pool_miss clusters point to context (add the missing STT phrases to the bank), routing (more sources or a larger k) \
or a larger pool_cap. ranked_wrong clusters point to routing (k, the closed call, the Gemini gate) or tools (a noisy \
feature off, viseme costs for look-alike shapes). abstained clusters point to the ∅ features (none_closed, \
no_speech, bad_quality) or the prior strength.
</guidance>

<example>
{{"pattern": "ranked_wrong, 41 failures; top confusions 'call mom' -> 'play some music' (6)", "hypothesis": "With \
k=1 the vote and top_vote features come from a single noisy sample, so a wrong first hypothesis wins; k=3 lets the \
true phrase win on agreement across samples, which should lift held-out word accuracy.", "change": {{"op": \
"set_k", "k": 3}}, "failures_cited": ["clip-0012", "clip-0031"]}}
</example>

<self_check>
Before answering, check each Required item against the inputs above; if any fails, pick a different change.
</self_check>"""


def build_prompt(context: dict) -> list[dict]:
    summary = context["failures"]
    user = USER_PROMPT.format(
        features=", ".join(FEATURES), train_clips=summary["train_clips"], n_failures=summary["failures"],
        failures=json.dumps({k: summary[k] for k in ("clusters", "examples")}, indent=1),
        current_id=context["current"]["id"], config=json.dumps(context["config"]),
        train_metrics=json.dumps(context["current"]["train"]), memory=json.dumps(context["memory"], indent=1),
        change_space=json.dumps(context["change_space"], indent=1))
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


class MuseProposer:
    """Muse Spark over Chat Completions with json_schema structured output (temperature 1.0, effort low). The
    proposal is checked with the loop's own `check`; an invalid one is sent back once with the reason."""

    def __init__(self, client=None, model: str = MODEL):
        self.client, self.model = client, model

    def propose(self, context: dict, check) -> dict:
        from muse_sampler import _wire, from_wire  # ordinal key prefixes: Meta generates keys alphabetically
        from muse_video import make_client

        client = self.client or make_client()
        schema = _wire(proposal_schema(context), start=1)
        messages = build_prompt(context)
        cost, proposal = 0.0, None
        for attempt in (1, 2):
            response = client.chat.completions.create(
                model=self.model, temperature=1.0, reasoning_effort="low", messages=list(messages),
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "one_change", "schema": schema, "strict": True}})
            usage = getattr(response, "usage", None)
            cost += ((getattr(usage, "prompt_tokens", 0) or 0) * MUSE_PRICE[0]
                     + (getattr(usage, "completion_tokens", 0) or 0) * MUSE_PRICE[1]) / 1e6
            content = response.choices[0].message.content
            raw = from_wire(json.loads(content))
            proposal = {"pattern": raw.get("pattern"), "hypothesis": raw.get("hypothesis", ""),
                        "change": raw.get("change"), "failures_cited": raw.get("failures_cited") or []}
            try:
                check(proposal)
                break
            except InvalidChange as e:
                messages += [{"role": "assistant", "content": content},
                             {"role": "user", "content": f"That change was refused: {e}. Propose a different ONE "
                                                         "change from the change space, in the same JSON format."}]
        return {**proposal, "cost_usd": cost, "attempts": attempt}


# ---------------------------------------------------------------- the loop


class _NoTest:
    """The loop's view of the evaluator: the test split is refused (only `final` may evaluate it)."""

    def __init__(self, inner):
        self.inner = inner

    def evaluate(self, records, config, split_name, limit=None):
        if split_name == "test":
            raise RuntimeError("the loop never evaluates the test split; use final()")
        if limit is None:
            return self.inner.evaluate(records, config, split_name)
        return self.inner.evaluate(records, config, split_name, limit=limit)


def _paid_config(config: dict) -> bool:
    return bool(config["prompt_hint"])  # stored raw outputs were recorded without a hint; a hint needs re-calls


def _charge(metrics: dict, config: dict) -> float:
    return metrics["clips"] * (metrics["cost_per_clip"] or 0.0) if _paid_config(config) else 0.0


class _Loop:
    def __init__(self, records, registry, evaluator, *, budget, free_only, log):
        self.records, self.registry, self.free_only, self.log = records, registry, free_only, log
        self.evaluator = evaluator
        self.ev = _NoTest(evaluator)
        self.budget = budget
        self.train = split(records, "train")
        self.n_heldout = len(split(records, "heldout"))
        self.digest = records_digest(records)
        self.spent = 0.0
        self.refused: list[dict] = []

    # -------- helpers

    def _metrics(self, vid: str, split_name: str, config: dict) -> dict:
        metrics = self.ev.evaluate(self.records, config, split_name)
        self.registry.record_metrics(vid, split_name, metrics)
        return metrics

    def _current(self) -> tuple[str, dict]:
        vid = self.registry.pointers()["current"]
        return vid, self.registry.get(vid)

    def _step(self, iteration, version, change, hypothesis, decision, reason, heldout=None) -> dict:
        current_id, current = self._current()
        kind = None
        try:
            kind = kind_of(change)
        except InvalidChange:
            pass
        step = {"iteration": iteration, "version": version, "kind": kind, "change": change, "hypothesis": hypothesis,
                "decision": decision, "reason": reason, "heldout_word_acc": heldout, "current": current_id,
                "current_word_acc": current["metrics"]["heldout"]["word_acc"]}
        self.log(f"[{iteration}] {decision:8s} {version or '-':8s} {_describe(change)} — {reason}")
        return step

    def ensure_baseline(self, baseline: dict | None) -> None:
        versions = self.registry.versions()
        if not versions:
            self.registry.create_baseline(baseline if baseline is not None else default_config())
        elif baseline is not None and config_hash(baseline) != versions[0]["config_hash"]:
            raise ValueError("this registry was started from a different baseline config")
        vid, current = self._current()
        if current["metrics"]["heldout"] is None:  # first start, or a crash before the baseline was scored
            config = self.registry.config(vid)
            self._metrics(vid, "train", config)
            self._metrics(vid, "heldout", config)

    def check(self, config: dict, code: str | None = None):
        versions = self.registry.versions()
        unsupported = getattr(self.evaluator, "unsupported", None)

        def check(proposal) -> dict:
            if not isinstance(proposal, dict) or "change" not in proposal:
                raise InvalidChange("a proposal needs a change")
            change = validate_change(proposal["change"], config, free_only=self.free_only)
            for v in versions:
                if v["status"] == "rejected" and v["change"] == change:
                    raise InvalidChange(f"already tried and rejected as {v['id']}: {v['reason']}")
            new = apply(config, change)
            new_hash = version_hash(new, code)
            for v in versions:
                if v["config_hash"] == new_hash:
                    raise InvalidChange(f"the resulting config was already evaluated as {v['id']} ({v['status']})")
            if unsupported and (why := unsupported(new)):
                raise InvalidChange(f"the evaluator cannot run this change: {why}")
            return change

        return check

    # -------- evaluating one candidate (never judging it)

    def _evaluated(self, iteration, vid: str, note: str | None = None) -> dict:
        """Attach the evidence and leave the version unjudged; the step's reason is the evidence in one line."""
        record = self.registry.get(vid)
        parent = self.registry.get(record["parent"])
        mine, theirs = record["metrics"], parent["metrics"]
        deltas = {split_name: heldout_delta(theirs[split_name], mine[split_name])
                  if mine[split_name] and theirs[split_name] else None for split_name in ("train", "heldout")}
        evidence = {"by": "iterate.py", "parent": parent["id"], "records": self.digest,
                    "train": mine["train"], "heldout": mine["heldout"], "parent_train": theirs["train"],
                    "parent_heldout": theirs["heldout"], "train_delta_pts": deltas["train"],
                    "heldout_delta_pts": deltas["heldout"], "pilot": record["pilot"], "note": note}
        self.registry.evaluated(vid, evidence)
        held = deltas["heldout"]
        reason = note or (f"held-out word_acc {held['word_acc']:+.2f} pts, viseme_acc {held['viseme_acc']:+.2f} pts "
                          f"vs {parent['id']}; unjudged")
        return self._step(iteration, vid, record["change"], record["hypothesis"], "evaluated", reason,
                          mine["heldout"]["word_acc"] if mine["heldout"] else None)

    def evaluate(self, vid: str, iteration) -> dict:
        record = self.registry.get(vid)
        config, parent_config = self.registry.config(vid), self.registry.config(record["parent"])

        def spend(usd):
            self.spent += usd
            self.registry.add_cost(vid, usd)

        if _paid_config(config) and record["pilot"] is None:  # spend guards: they skip spending, never judge
            try:
                pilot = self.ev.evaluate(self.records, config, "heldout", limit=PILOT_CLIPS)
                spend(_charge(pilot, config))
                base = self.ev.evaluate(self.records, parent_config, "heldout", limit=PILOT_CLIPS)
                spend(_charge(base, parent_config))
            except PilotUnsupported as e:
                return self._evaluated(iteration, vid, f"paid change but no pilot possible: {e}; nothing measured")
            self.registry.set_pilot(vid, {"candidate": pilot, "parent": base})
            if _pts(pilot["word_acc"] - base["word_acc"]) <= 0:
                return self._evaluated(iteration, vid, (
                    f"pilot on {pilot['clips']} held-out clips: word_acc {100 * base['word_acc']:.1f} -> "
                    f"{100 * pilot['word_acc']:.1f}, no gain; full evaluation skipped to save money"))
            full = (len(self.train) + self.n_heldout) * (pilot["cost_per_clip"] or 0.0)
            if self.spent + full > self.budget:
                return self._evaluated(iteration, vid, f"pilot improved, but the full evaluation (~${full:.2f}) "
                                                       "exceeds the budget; full evaluation skipped")

        if record["metrics"]["train"] is None:
            spend(_charge(self._metrics(vid, "train", config), config))
        spend(_charge(self._metrics(vid, "heldout", config), config))
        return self._evaluated(iteration, vid)

    # -------- one proposal

    def refuse(self, iteration, proposal, reason) -> dict:
        change = proposal.get("change") if isinstance(proposal, dict) else proposal
        kind = None
        try:
            kind = kind_of(change)
        except InvalidChange:
            pass
        self.refused.append({"kind": kind, "change": change, "hypothesis": (proposal or {}).get("hypothesis", ""),
                             "reason": reason})
        return self._step(iteration, None, change, (proposal or {}).get("hypothesis", ""), "refused", reason)

    def iterate(self, iteration, proposer) -> dict | None:
        current_id, current = self._current()
        config = self.registry.config(current_id)
        summary = summarize(self.evaluator.failures(self.train, config), len(self.train))
        context = {
            "iteration": iteration,
            "current": {"id": current_id, "train": current["metrics"]["train"]},
            "config": config,
            "failures": summary,
            "memory": tried_ideas(self.registry.versions(), self.refused),
            "change_space": change_space(self.free_only),
            "free_only": self.free_only,
            "budget_left": None if math.isinf(self.budget) else round(self.budget - self.spent, 4),
        }
        check = self.check(config, self.registry.code(current_id))
        proposal = proposer.propose(context, check)
        if proposal is None:
            return None
        self.spent += proposal.get("cost_usd", 0.0)
        try:
            change = check(proposal)
        except InvalidChange as e:
            return self.refuse(iteration, proposal, f"invalid: {e}")
        if is_paid(change):
            per_clip = current["metrics"]["heldout"]["cost_per_clip"]
            if per_clip is None:
                return self.refuse(iteration, proposal, "paid change, but the cost per clip is unknown")
            pilot = min(PILOT_CLIPS, self.n_heldout) * per_clip
            if self.spent + pilot > self.budget:
                return self.refuse(iteration, proposal, f"over budget: the pilot needs ~${pilot:.2f}, "
                                                        f"${max(0.0, self.budget - self.spent):.2f} left")
        shown = {e["id"] for e in summary["examples"]}
        candidate = self.registry.create_candidate(
            current_id, change, hypothesis=proposal.get("hypothesis", ""), author="agent",
            failures_cited=[c for c in proposal.get("failures_cited") or [] if c in shown],
            cost_usd=proposal.get("cost_usd", 0.0))
        return self.evaluate(candidate["id"], iteration)


def run(records: list[dict], registry, proposer, evaluator, *, baseline: dict | None = None, iterations: int = 20,
        budget: float = math.inf, free_only: bool = False, log=None) -> list[dict]:
    """Evaluate up to `iterations` proposals, each ONE change on the registry's current version, and record each as
    `evaluated` with its evidence; returns one step per proposal (and one per unfinished candidate resumed). Moves no
    pointer. Stops early when the proposer has nothing left or when this run has spent `budget` dollars."""
    loop = _Loop(records, registry, evaluator, budget=budget, free_only=free_only, log=log or (lambda line: None))
    loop.ensure_baseline(baseline)
    steps = [loop.evaluate(v["id"], "resume") for v in registry.versions() if v["status"] == "candidate"]
    start = len(registry.versions()) - 1
    for i in range(1, iterations + 1):
        if loop.spent >= budget:
            loop.log(f"stop: budget spent (${loop.spent:.2f} of ${budget:.2f})")
            break
        step = loop.iterate(start + i, proposer)
        if step is None:
            loop.log("stop: the proposer has no more ideas")
            break
        steps.append(step)
    return steps


def final(records: list[dict], registry, evaluator, version: str | None = None) -> dict:
    """Evaluate `version` (default: current) on the test split and record it: once per version, never again. The
    only path to `test`."""
    vid = version or registry.pointers()["current"]
    if registry.get(vid)["metrics"]["test"] is not None:
        raise RuntimeError(f"{vid} was already evaluated on the test split; the test split is read once")
    metrics = evaluator.evaluate(records, registry.config(vid), "test")
    registry.record_metrics(vid, "test", metrics)
    return metrics


def history(registry) -> list[dict]:
    """Staircase rows rebuilt from the registry (every version after the baseline, in order)."""
    versions = registry.versions()
    rows, current = [], versions[0] if versions else None
    for v in versions[1:]:
        if v["status"] == "adopted":
            current = v
        heldout = v["metrics"]["heldout"]
        rows.append({"iteration": v["seq"] - 1, "version": v["id"], "kind": v["kind"], "change": v["change"],
                     "hypothesis": v["hypothesis"], "decision": v["status"], "reason": v["reason"],
                     "heldout_word_acc": heldout["word_acc"] if heldout else None, "current": current["id"],
                     "current_word_acc": (current["metrics"]["heldout"] or {}).get("word_acc")})
    return rows


def staircase(steps: list[dict]) -> str:
    """The version staircase as a text table: iteration, kind, change, held-out word_acc, decision, current."""
    def pct(x):
        return "    -" if x is None else f"{100 * x:5.1f}"

    lines = [f"{'it':>6}  {'version':8}  {'kind':8}  {'change':44}  {'held-out':>8}  {'decision':8}  current",
             "-" * 104]
    for s in steps:
        lines.append(f"{s['iteration']!s:>6}  {s['version'] or '-':8}  {s['kind'] or '-':8}  "
                     f"{_describe(s['change'])[:44]:44}  {pct(s['heldout_word_acc']):>8}  {s['decision']:8}  "
                     f"{pct(s['current_word_acc'])} {s['current']}")
    return "\n".join(lines)
