# Pipeline

Green = built and verified · amber = being built · grey dashed = planned. The app's Pipeline page (`/pipeline.html`) renders this file live; edit it and reload.

## Runtime: what happens to one clip

```mermaid
flowchart TD
  CAM["Webcam in browser<br/>request 30 fps, show actual"]
  TRK["Classical tracker, every frame, ~1-2 ms<br/>NCC on nose-to-chin anchor, lips masked + One-Euro smoothing"]
  TRIG{"Lost, big motion, scale jump over 15%,<br/>or 4 s since last anchor?"}
  SAM["SAM 3.1 re-anchor on 'face'<br/>$0.0025 per call"]
  CROP["Muted mouth crop, scaled up"]
  QG{"Quality gate: too many<br/>occluded or blurred frames?"}
  REPEAT["Ask user to repeat<br/>free, before any model spend"]
  GEM["Gemini 4 Argon @ 24 fps, its max<br/>enum + logprobs → P(phrase)"]
  MUSEK["Muse Spark ×K open samples, temp 1.0<br/>schema v1: quality → visemes → word slots → hypotheses<br/>frame-numbered grids @ 24 fps · K=3 runtime, 5 preheat"]
  MUSEC["Muse Spark closed call<br/>score phrase bank + 'none of these'"]
  POOL["Candidate pool<br/>hypotheses, slot recombinations, closed + Gemini top,<br/>viseme neighbours from the bank, ∅"]
  FEAT["Features per candidate<br/>votes across K, self-scores, slot support, viseme distance,<br/>closed, Gemini, user prior, quality → ∅"]
  FUSE["Trainable scorer<br/>staged conditional logit → P(candidate)"]
  GATE{"P(top) ≥ learned threshold?"}
  SEND["Auto-send phrase"]
  CONFIRM["Ask user to confirm"]

  CAM --> TRK --> TRIG
  TRIG -- yes --> SAM -- new anchor --> TRK
  TRIG -- no --> CROP --> QG
  QG -- bad clip --> REPEAT
  QG -- ok --> GEM & MUSEK & MUSEC
  GEM & MUSEK & MUSEC --> POOL --> FEAT --> FUSE --> GATE
  GATE -- high --> SEND
  GATE -- low --> CONFIRM

  classDef built fill:#d1fae5,stroke:#059669,color:#064e3b
  classDef building fill:#fef3c7,stroke:#d97706,color:#78350f
  classDef planned fill:#f3f4f6,stroke:#9ca3af,color:#374151,stroke-dasharray:4 3
  class CAM,GEM,GATE,SAM,TRK,TRIG,QG,REPEAT built
  class CROP,MUSEK,MUSEC,POOL,FEAT,FUSE building
  class SEND,CONFIRM planned
```

## Learning: how it gets better

```mermaid
flowchart TD
  CLIP["Each recorded clip<br/>raw K samples + closed + Gemini outputs<br/>features recomputed from raw: replays are free"]
  STT["STT on: Muse Voice Transcribe<br/>reference label"]
  CONF["Video only: the user's confirm<br/>answer is a label too"]
  MATCH["Transcript → phrase ID"]
  STORE[("Clip store<br/>MongoDB Atlas, local fallback")]

  PREQ["Predict, then train<br/>honest accuracy: video-only vs STT"]
  UI["Preheat UI<br/>accuracy, learning curve, stage, camera fps"]
  STAGES["Staged training by label count<br/>0 defaults · 10 temperature · 30 core features · 60 all features"]
  FUSE["Fusion weights + auto-send threshold"]

  GT["SAM video ground truth<br/>$0.20 per 1,000 frames"]
  REPLAY["Tracker replay + knob sweep<br/>free"]
  KNOBS["Tracker thresholds"]

  AGENT["Muse Spark agent reviews STT disagreements<br/>keeps a change only if replayed accuracy rises"]
  PROMPTS["Few-shot clips, prompt hints, phrase bank,<br/>viseme substitution costs, new features"]

  CLIP --> STORE
  STT --> MATCH --> STORE
  CONF --> STORE
  STORE --> PREQ --> UI
  STORE --> STAGES --> FUSE
  STORE --> GT --> REPLAY --> KNOBS
  STORE --> AGENT --> PROMPTS

  classDef built fill:#d1fae5,stroke:#059669,color:#064e3b
  classDef building fill:#fef3c7,stroke:#d97706,color:#78350f
  classDef planned fill:#f3f4f6,stroke:#9ca3af,color:#374151,stroke-dasharray:4 3
  class STT,MATCH,GT,STORE,UI,REPLAY,KNOBS built
  class CLIP,STAGES,PREQ,FUSE building
  class CONF,AGENT,PROMPTS planned
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
