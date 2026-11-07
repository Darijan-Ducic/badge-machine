# Status — 2026-11-07

## Goal
Lip reading from a laptop webcam (VTT = video-to-text), trained passively from everyday dictation (STT = speech-to-text
is the label), demonstrating the base problem in `docs/self-improvement.md`: a system that improves itself, each change
versioned and adopted only if held-out word accuracy rises. **Next milestone: run 1** on webcam-emulated public-domain
Obama Weekly Addresses.

## Built and verified (on `main`)
| Piece | Where | Evidence |
|---|---|---|
| Live v1 clip path: STT or given transcript → Muse open ×K (schema v1) + Muse closed + Gemini in parallel → raw outputs + usage + config stored | `clips.py`, `server.py` | 247 py tests; live smoke vs Atlas |
| Crop: median of tracker boxes → muted H.264, 256 px (`v1-tracker-median`); SAM-face fallback (`v1-static-median`) | `crop.py` | tests; live smoke |
| Muse sampler: K parallel schema-v1 samples in reasoning order (ordinal wire keys), frame-numbered 2×2 grids @ 24 fps, closed call scoring phrases then "none" | `muse_sampler.py` | live: 3/3 valid; key order verified by streaming |
| Gemini 4 Argon closed scores via enum + logprobs @ 24 fps (its max) | `scorers.py` | live |
| STT: Muse Voice Transcribe (Spark 1.3 dropped audio ~50%) | `server.py` | live |
| Visemes-v1 (CMUdict) + weighted viseme distance, editable costs | `visemes.py`, `data/viseme_costs.json` | 59 tests, 13/13 mutants |
| Candidate pool + 13 features + ∅, exact text matching (fuzzy merged "turn on/off") | `candidates.py` | tests |
| Staged scorer (defaults → temperature → core → full), masked conditional logit, out-of-fold auto-send threshold | `scorer.py`, `fusion.py` | tests, 21/21 mutants |
| SAM 3.1 client + `/sam`; SAM video ground-truth tool | `sam.py`, `tools/sam_groundtruth.py` | live |
| Browser tracker (NCC anchor, lips masked, One-Euro, SAM re-anchor every 4 s, quality gate) + replay/sweep | `static/tracker.js`, `tools/replay.mjs` | 16 node tests; 0.035 face-width error, 0.33 SAM calls/s, ~2 ms/frame on a real clip |
| Clip store on MongoDB Atlas (records + GridFS), file fallback | `clips.py` | live round-trip |
| Practice ingest (seeded, resumable) | `tools/ingest_practice.py` | live smoke |
| App: Record, Preheat, Dataset, Clips (VTT vs STT diff), Stats (curve + training strip), Training (v1 sheet, Self-improvement, Runs), Pipeline (live diagrams), Tracker | `static/` | headless Chrome, stubs for not-yet-wired endpoints |

## Measured
- **Cost ≈ $0.11/clip** at K=5 (Muse $0.082 — mostly reasoning output tokens; Gemini $0.012; SAM $0.018). STT $0.18/hour.
- Latency per clip ≈ 42 s end to end (Muse open 14–22 s, closed 24 s, Gemini 6 s, parallel).
- Research: lip reading degrades below ~21 fps; per-user command accuracy plateaus at 3–5 examples per phrase (LipLearner, CHI 2023).

## In progress (subagents, each in its own worktree)
| Agent | Scope |
|---|---|
| `sessions` | `asr.py` (turns), `sessions.py`: dictation session → STT turns → clips, `/sessions` contract |
| `lipmode` | `static/capture.js` (camera + tracker + full-frame recording + per-frame boxes), `lipmode.html`, then `dictation.html` |
| `webcam-emu` | Obama footage → fake webcam/mic (640×480 4:3, face 30% width) → driven through the real Dictation page |
| `replay-lab` | `experiment.py`, `runs.py`, `readiness.py`: free replays, splits 60/20/20, word/sentence/viseme/top-3 accuracy, cost |
| `iterator` | `versions.py`, `iterate.py`: version registry (`live` vs `current`), autonomous loop, held-out gate |

## Path to run 1
1. Merge the five agents above as they land (verify scope, removed assertions, tests, screenshots).
2. Wire the server: `POST /dictation`, `GET /sessions[/<id>[/video]]`, `GET /versions`, `GET /runs`, `GET /readiness`, `POST /vtt`, `POST /vtt/<id>/label`.
3. Pilot: one emulated session (~10 turns, ≈ $1.10) through `dictation.html`; check it in Dataset → Clips → Stats.
4. User picks the full run size (150 utterances ≈ $17, 300 ≈ $33 at measured cost).
5. Run 1 → autonomous iterations (free changes first) → Self-improvement staircase.

## Decisions (locked)
- Natural preheat (no prompted phrases); phrase bank learned from STT history; lip-only readiness per phrase.
- Rich schema v1 for VTT; K=5 for dictation/practice, K=3 runtime; temperature 1.0, effort low; closed call separate so the list can't anchor the open answers.
- One crop path for training and runtime: browser tracker boxes → server-side median crop.
- Raw outputs are immutable; features are recomputed, so post-processing changes replay for free.
- Performance is versioning: every chart point is a version; `current` moves on the gate, `live` only with user approval.
- Hosted APIs only — no Docker (`CLAUDE.md`).

## Open questions
- Should free (post-processing) changes auto-promote to `live` when the gate passes?
- Full run size / budget cap.
- Mouth crop sits right of centre (`MOUTH_CX = .5`): first system-loop experiment candidate (tracker knob vs SAM lips ground truth).
- Static per-clip crop vs moving crop under fast head motion (free experiment once data exists).

## Known issues
- `scorer.prequential` marks unlabeled clips as pool hits (callers pass labeled clips only) — fix with a test.
- Clip detail has no `vtt_muse` field (the UI derives it) — add server-side.
- Gemini key and Atlas password were shared in chat — rotate after the hackathon.
- 13 merged agent worktrees/branches remain under `.claude/worktrees/` — delete only with approval.

## Run it
- `uv run --env-file .env server.py` → http://localhost:8765 (needs `MODEL_API_KEY`, `GEMINI_API_KEY`, `MONGODB_URI`).
- Tests: `uv run pytest -q` and `node --test tests/*.test.mjs`.

## Docs
`docs/pipeline.md` (all diagrams, live on the Pipeline page) · `docs/training-v1.md` (approval sheet) ·
`docs/self-improvement.md` (problem statement mapping, metrics) · `docs/uncertainty-design.md` (schema, features,
contracts) · `CLAUDE.md` (project rules).
