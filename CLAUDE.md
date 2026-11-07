# badge-machine

## Pipeline diagram
- `docs/pipeline.md` is the single source of truth for the pipeline diagrams; the app's Pipeline page renders it live (served at `/pipeline.md`), so editing the file and reloading is the whole update.
- ALWAYS update `docs/pipeline.md` (stage + build-status `class` line) in the same commit that adds, removes, or changes a pipeline stage — the diagrams change often and the team reads them as the current state.
- A new `## heading` followed by a ```mermaid block becomes a new button on the Pipeline page; no page change needed.
- ALWAYS validate before committing, because a broken block shows an error on the page:
  `awk '/^```mermaid/{n++;f=1;next}/^```/{f=0}f{print > ("/tmp/pipeline-" n ".mmd")}' docs/pipeline.md && for f in /tmp/pipeline-*.mmd; do mmdc -q -i "$f" -o /tmp/check.svg || echo "BROKEN: $f"; done`
