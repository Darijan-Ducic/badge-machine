# badge-machine

## Infrastructure
- NEVER use Docker or run local infrastructure (no local MongoDB, no containers) — everything is hosted APIs: Meta Model API (Muse Spark, SAM, Muse Voice Transcribe), Gemini, MongoDB Atlas (`MONGODB_URI` in `.env`). Test external stores with in-process fakes plus a live check against the hosted service.

## Naming
- VTT = video-to-text (lip-reading output): the combined scorer's pick is "VTT", per model "VTT · Gemini" / "VTT · Muse". STT = speech-to-text, the label. ALWAYS use these names in the UI, docs, and new API fields.

## Frontend visibility
- ALWAYS surface every curated dataset, session, turn, clip, and training run in the app (Dataset / Clips / Stats / Training pages) — the user watches training through the UI, never only via CLI output.

## Pipeline diagram
- `docs/pipeline.md` is the single source of truth for the pipeline diagrams; the app's Pipeline page renders it live (served at `/docs/pipeline.md` via the `static/docs` symlink; the Training page renders `docs/training-v1.md` the same way), so editing the file and reloading is the whole update.
- ALWAYS update `docs/pipeline.md` (stage + build-status `class` line) in the same commit that adds, removes, or changes a pipeline stage — the diagrams change often and the team reads them as the current state.
- A new `## heading` followed by a ```mermaid block becomes a new button on the Pipeline page; no page change needed.
- ALWAYS validate before committing, because a broken block shows an error on the page:
  `awk '/^```mermaid/{n++;f=1;next}/^```/{f=0}f{print > ("/tmp/pipeline-" n ".mmd")}' docs/pipeline.md && for f in /tmp/pipeline-*.mmd; do mmdc -q -i "$f" -o /tmp/check.svg || echo "BROKEN: $f"; done`

## Self-improvement
- The iterating agent is a Claude Code session, not a hard-coded loop — it reads VTT vs STT results, decides what to
  change (code, prompts, features, costs, thresholds), and records each change as a version in `versions.py`.
- ALWAYS add iteration capability as an agent-usable script that reports evidence (paired per-clip wins/losses,
  held-out delta, cost) plus a section in the skill book; NEVER as a new knob enum, fixed adopt threshold, or
  in-code proposer. The agent never tunes on the test split.
- Update the skill book in the same commit as any change to the iteration scripts.
