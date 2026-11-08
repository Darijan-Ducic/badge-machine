# Pipeline

Green = built and verified · amber = being built · grey dashed = planned. The app's Pipeline page (`/pipeline.html`) renders this file live; edit it and reload.

**Who updates:** every iteration, system and personal, is run by a local Claude Code session following the skill book. Scripts are its instruments and report evidence; the session decides and versions each change. Nothing learns by itself in code. Muse agents are reserved for the product, later.

## Runtime: what happens to one clip

```mermaid
flowchart TD
  CAM["Webcam in browser<br/>request 30 fps, show actual"]
  TRK["Classical tracker, every frame, ~1-2 ms<br/>NCC on nose-to-chin anchor, lips masked + One-Euro smoothing"]
  TRIG{"Lost, big motion, scale jump over 15%,<br/>or 4 s since last anchor?"}
  SAM["SAM 3.1 re-anchor on 'face'<br/>$0.0025 per call"]
  CROP["Muted mouth crop, 256 px<br/>server-side median of the tracker's boxes<br/>identical for dictation and lip mode"]
  QG{"Quality gate: too many<br/>occluded or blurred frames?"}
  REPEAT["Ask user to repeat<br/>free, before any model spend"]
  GEM["Gemini 4 Argon @ 24 fps, its max<br/>enum + logprobs → P(phrase)"]
  MUSEK["Muse Spark ×K open samples, temp 1.0<br/>schema v1: quality → visemes → word slots → hypotheses<br/>frame-numbered grids @ 24 fps · K=3 runtime, 5 preheat"]
  MUSEC["Muse Spark closed call<br/>score phrase bank + 'none of these'"]
  POOL["Candidate pool<br/>hypotheses, slot recombinations, closed + Gemini top,<br/>viseme neighbours from the bank, ∅<br/>closed, Gemini and bank slots reserved"]
  SENT["Sentence path → VTT, dictation turns<br/>reading longer than every phrase:<br/>ROVER word vote over the K top hypotheses"]
  FEAT["Features per candidate<br/>votes across K, self-scores, slot support, viseme distance,<br/>closed, Gemini, user prior, quality → ∅"]
  FUSE["Trainable scorer → VTT<br/>staged conditional logit → P(candidate)"]
  GATE{"P(top) ≥ learned threshold?"}
  SEND["Auto-send phrase"]
  CONFIRM["Ask user to confirm"]

  CAM --> TRK --> TRIG
  TRIG -- yes --> SAM -- new anchor --> TRK
  TRIG -- no --> CROP --> QG
  QG -- bad clip --> REPEAT
  QG -- ok --> GEM & MUSEK & MUSEC
  GEM & MUSEK & MUSEC --> POOL --> FEAT --> FUSE --> GATE
  MUSEK -- sentence turn --> SENT
  GATE -- high --> SEND
  GATE -- low --> CONFIRM

  classDef built fill:#d1fae5,stroke:#059669,color:#064e3b
  classDef building fill:#fef3c7,stroke:#d97706,color:#78350f
  classDef planned fill:#f3f4f6,stroke:#9ca3af,color:#374151,stroke-dasharray:4 3
  class CAM,GEM,GATE,SAM,TRK,TRIG,QG,REPEAT,CROP,MUSEK,MUSEC,POOL,FEAT,FUSE,SEND,CONFIRM,SENT built
```

## Learning: how it gets better

```mermaid
flowchart TD
  CLIP["Each recorded clip<br/>raw K samples + closed + Gemini outputs<br/>features recomputed from raw: replays are free"]
  STT["STT on: dictation cut at our own pauses<br/>2–8 s turns, Muse Voice Transcribe per turn<br/>reference label"]
  CONF["Video only: the user's confirm<br/>answer is a label too"]
  MATCH["Transcript → phrase ID"]
  STORE[("Clip store<br/>MongoDB Atlas, local fallback")]

  PREQ["Predict, then train<br/>honest accuracy: VTT vs STT"]
  UI["App: Dataset · Clips · Stats<br/>VTT vs STT per clip, learning curve"]
  STAGES["Staged training by label count<br/>0 defaults · 10 temperature · 30 core features · 60 all features"]
  FUSE["Fusion weights + auto-send threshold"]

  GT["SAM video ground truth<br/>$0.20 per 1,000 frames"]
  REPLAY["Tracker replay + knob sweep<br/>free"]
  KNOBS["Tracker thresholds"]

  PAGENT["By: Claude Code session, local · skill book<br/>personal: decides when to re-fit,<br/>checks the user's newest clips, versions it"]
  AGENT["By: Claude Code session, local · skill book<br/>system: reviews STT disagreements,<br/>keeps a change only if held-out accuracy rises"]
  PROMPTS["Few-shot clips, prompt hints, phrase bank,<br/>viseme substitution costs, new features, code"]

  CLIP --> STORE
  STT --> MATCH --> STORE
  CONF --> STORE
  STORE --> PREQ --> UI
  STORE --> PAGENT --> STAGES --> FUSE
  STORE --> GT --> REPLAY --> AGENT
  STORE --> AGENT --> PROMPTS & KNOBS

  classDef built fill:#d1fae5,stroke:#059669,color:#064e3b
  classDef building fill:#fef3c7,stroke:#d97706,color:#78350f
  classDef planned fill:#f3f4f6,stroke:#9ca3af,color:#374151,stroke-dasharray:4 3
  class STT,MATCH,GT,STORE,UI,REPLAY,KNOBS,CLIP,STAGES,PREQ,FUSE built
  class PROMPTS building
  class AGENT building
  class CONF,PAGENT planned
```

## Personal loop: learns from each user's own dictation

```mermaid
flowchart LR
  USE["User dictates normally<br/>STT on, no prompted phrases"]
  LBL["Label = STT turn text<br/>video-only mode: confirm/edit = label"]
  BANK["Phrase bank + frequencies<br/>feeds the prior feature"]
  W["Scorer weights, staged<br/>0 → 10 → 30 → 60 labeled clips"]
  TH["Auto-send threshold<br/>out-of-fold, precision 0.9"]
  FEW["Few-shot examples<br/>user's own best clips in prompts"]
  READY["Per-phrase readiness<br/>lip-only mode unlocks per phrase"]
  CAL["Camera framing calibration<br/>face width, mouth ratios"]
  OUT["Opt-in clips → system replay sets"]
  IN["New system version → re-fit<br/>from stored raw outputs"]
  AG["By: Claude Code session, local · skill book<br/>reads this user's VTT vs STT, decides when to re-fit,<br/>runs the fit scripts, versions each personal model"]

  USE --> LBL
  USE --> CAL
  LBL --> AG
  AG --> BANK & W & TH & FEW & CAL
  W & TH --> READY
  LBL -.-> OUT
  IN -.-> AG

  classDef built fill:#d1fae5,stroke:#059669,color:#064e3b
  classDef building fill:#fef3c7,stroke:#d97706,color:#78350f
  classDef planned fill:#f3f4f6,stroke:#9ca3af,color:#374151,stroke-dasharray:4 3
  class LBL,BANK,W,TH,USE built
  class IN,READY building
  class FEW,CAL,OUT,AG planned
```

## System loop: what we iterate, versioned v1 → v2

```mermaid
flowchart LR
  SETS["Replay sets<br/>practice footage + opt-in user clips<br/>raw model outputs stored"]
  ERR["Review errors<br/>STT vs lip disagreements"]
  AGENT["By: Claude Code session, local · skill book<br/>proposes ONE change, runs the replay scripts"]
  PAID["Call-level pieces<br/>re-calls the models, costs money<br/>───<br/>output schema + prompts<br/>grid · fps · crop<br/>K · temperature · effort"]
  FREE["Post-processing pieces<br/>replay is free<br/>───<br/>viseme substitution costs<br/>candidate pool rules + features<br/>stage plan + cold-start weights<br/>tracker knobs vs SAM ground truth"]
  GATE{"Session judges the evidence:<br/>held-out accuracy up on the same clips?"}
  VER["New system version<br/>config hash on every clip"]
  REFIT["Every user's personal layers re-fit<br/>by the session, from stored raw outputs"]

  SETS --> ERR --> AGENT
  AGENT --> PAID & FREE
  PAID & FREE --> GATE
  GATE -- yes --> VER --> REFIT
  GATE -- "no: revert" --> AGENT

  classDef built fill:#d1fae5,stroke:#059669,color:#064e3b
  classDef building fill:#fef3c7,stroke:#d97706,color:#78350f
  classDef planned fill:#f3f4f6,stroke:#9ca3af,color:#374151,stroke-dasharray:4 3
  class PAID,FREE,VER built
  class SETS,REFIT,ERR,AGENT,GATE building
```

## Update cadence: when each piece learns

```mermaid
flowchart LR
  subgraph clip ["Every labeled clip · seconds · recorded, nothing learns"]
    direction TB
    C1["STT turn arrives → clip stored<br/>raw outputs + call-level config, never edited"]
    C2["Phrase bank count +1"]
    C1 --> C2
  end
  subgraph milestone ["Personal iteration · by the Claude Code session · when the skill book says"]
    direction TB
    C3["Re-fit scorer weights + threshold<br/>from this user's raw clips, < 1 s script<br/>session checks the newest clips, versions it"]
    M1["Stage unlocks more free weights<br/>defaults → temperature → core → full<br/>session decides; default plan 10 / 30 / 60 labels"]
    C4["Per-phrase readiness updated"]
    C3 --> M1 --> C4
  end
  subgraph offline ["System iteration · by the Claude Code session · nightly or on demand"]
    O1["Skill book drives the replay scripts<br/>free experiments run freely,<br/>paid ones only within a budget"]
  end
  subgraph promote ["On promotion · session judges held-out + your OK"]
    direction TB
    P1["System version bump<br/>sys-v1 → sys-v2"]
    P2["Every user's personal layers re-fit by the session<br/>free when only post-processing changed"]
    P1 --> P2
  end
  clip --> milestone
  clip -. "opt-in clips" .-> offline
  offline --> promote
  promote -. "new defaults for every clip after" .-> clip

  classDef built fill:#d1fae5,stroke:#059669,color:#064e3b
  classDef building fill:#fef3c7,stroke:#d97706,color:#78350f
  classDef planned fill:#f3f4f6,stroke:#9ca3af,color:#374151,stroke-dasharray:4 3
  class C1,C2 built
  class C3,C4,M1,O1,P1,P2 building
```

## Versioning: what every clip, model, and run records

```mermaid
erDiagram
  USER ||--o{ CLIP : records
  DATASET ||--|{ CLIP : contains
  CLIP ||--|{ SAMPLE : "K open + closed + Gemini"
  USER ||--o{ PERSONAL_MODEL : has
  SYSTEM_VERSION ||--o{ PERSONAL_MODEL : "re-fit under"
  SYSTEM_VERSION ||--o{ RUN : "evaluated by"
  DATASET ||--o{ RUN : "replayed in"

  CLIP {
    string id
    string user_id
    string session_id
    string call_config_hash "frozen when recorded"
    string stt_text "the label"
    string label_source "stt, confirm, edit"
    json tracker_boxes "per frame"
  }
  SAMPLE {
    string call "open, closed, gemini"
    int sample_idx
    json output "raw, never edited"
    int latency_ms
    json usage "tokens"
  }
  SYSTEM_VERSION ||--o{ SYSTEM_VERSION : "parent of"
  SYSTEM_VERSION {
    string id "sys-v1, sys-v2"
    string parent "the version it changed"
    string kind "skills, tools, routing, context"
    json change "the ONE change"
    json metrics "word, sentence, viseme, top-3 per split"
    string call_config "schema, prompt, grid, fps, crop, K, temperature, effort, models"
    string post_config "viseme costs, features, pool rules, stages, tracker knobs"
    string code_commit "git hash"
    string author "Claude Code session or human"
    string status "live, current, adopted, rejected"
  }
  PERSONAL_MODEL {
    string system_version
    int trained_on "labeled clips"
    string stage
    float threshold
    string author "Claude Code session"
  }
  RUN {
    string id
    string baseline_run
    string dataset_order "dataset id + order seed"
    string hypothesis "why this change"
    string author "Claude Code session or human"
    json metrics "VTT vs STT, held-out"
    string gate "pass or fail"
  }
  DATASET {
    string id
    string source "practice footage or opt-in users"
  }
```

A personal model is never stored as opaque weights: it is a deterministic re-fit of (system version, the user's raw
clips up to `trained_on`), chosen and recorded by the Claude Code session, so any version can be rebuilt, compared,
or rolled back.

## Agent iteration: how one change gets in

```mermaid
sequenceDiagram
  autonumber
  participant S as Clip store
  participant A as Claude Code session (local, skill book)
  participant L as Replay lab
  participant R as Runs log
  participant H as You
  participant V as Version registry
  S->>A: STT vs VTT disagreements, by phrase and quality
  A->>R: read past runs, skip ideas that already failed
  A->>A: hypothesis + ONE change
  alt free knob: post-processing
    A->>L: replay baseline vs variant, same clips, same order
  else paid knob: call-level
    A->>L: pilot on 20 clips within the budget
    L-->>A: pilot metrics
    A->>L: full re-run only if the pilot improves
  end
  L-->>R: run record: config hash, metrics, per-clip wins/losses
  R-->>A: evidence
  alt session judges: held-out VTT accuracy up
    A->>V: adopt: current → candidate version
    R-->>H: proposed promotion + curve vs baseline (Training page)
    H->>V: approve → live = current
    A->>S: re-fit every user's personal layers
  else not better
    R-->>A: logged as a failed idea, reverted
  end
```

## Stages at a glance

| Stage | Where | Cost per clip |
|---|---|---|
| Capture + tracking | Browser | Free; SAM only on re-anchor ($0.0025 each) |
| Quality gate | Browser | Free |
| Lip reading @ 24 fps | Gemini + Muse Spark, in parallel | ~4 s + ~14 s, run concurrently |
| Fusion + gate | Local Python | Free, milliseconds |
| STT label (STT-on only) | Muse Voice Transcribe | Per minute of audio |
| Training + accuracy | Local Python | Free |
