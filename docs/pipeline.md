# Pipeline

Green = built and verified · amber = being built · grey dashed = planned. Rendered copies (also shown in the app's Pipeline view): [runtime](../static/diagrams/runtime.svg), [learning](../static/diagrams/learning.svg).

## Runtime: what happens to one clip

```mermaid
flowchart TD
  CAM["Webcam in browser<br/>request 30 fps, show actual"]
  TRK["Classical tracker, every frame<br/>NCC on nose-to-chin anchor + Kalman smoothing"]
  TRIG{"Lost, big motion, scale jump over 15%,<br/>or 2 s since last anchor?"}
  SAM["SAM 3.1 re-anchor on 'face'<br/>$0.0025 per call"]
  CROP["Muted mouth crop, scaled up"]
  QG{"Quality gate: too many<br/>occluded or blurred frames?"}
  REPEAT["Ask user to repeat<br/>free, before any model spend"]
  GEM["Gemini 4 Argon @ 24 fps, its max<br/>enum + logprobs → P(phrase)"]
  MUSE["Muse Spark 1.3 @ 24 fps<br/>2x2 frame grids, ≤48 images<br/>self-reported P(phrase)"]
  FUSE["Fusion classifier<br/>calibrated P(phrase), local"]
  GATE{"P(top) ≥ learned threshold?"}
  SEND["Auto-send phrase"]
  CONFIRM["Ask user to confirm"]

  CAM --> TRK --> TRIG
  TRIG -- yes --> SAM -- new anchor --> TRK
  TRIG -- no --> CROP --> QG
  QG -- bad clip --> REPEAT
  QG -- ok --> GEM & MUSE
  GEM & MUSE --> FUSE --> GATE
  GATE -- high --> SEND
  GATE -- low --> CONFIRM

  classDef built fill:#d1fae5,stroke:#059669,color:#064e3b
  classDef building fill:#fef3c7,stroke:#d97706,color:#78350f
  classDef planned fill:#f3f4f6,stroke:#9ca3af,color:#374151,stroke-dasharray:4 3
  class CAM,GEM,MUSE,FUSE,GATE,SAM built
  class TRK,TRIG,CROP,QG,REPEAT building
  class SEND,CONFIRM planned
```

## Learning: how it gets better

```mermaid
flowchart TD
  CLIP["Each recorded clip<br/>mouth crop + Gemini and Muse scores"]
  STT["STT on: Muse Voice Transcribe<br/>reference label"]
  CONF["Video only: the user's confirm<br/>answer is a label too"]
  MATCH["Transcript → phrase ID"]
  STORE[("Clip store<br/>MongoDB Atlas, local fallback")]

  PREQ["Predict, then train<br/>honest accuracy: video-only vs STT"]
  UI["Preheat UI<br/>accuracy, learning curve, stage, camera fps"]
  STAGES["Staged training by label count<br/>0 none · 10 temperature · 30 per-model · 60 + phrase prior"]
  FUSE["Fusion weights + auto-send threshold"]

  GT["SAM video ground truth<br/>$0.20 per 1,000 frames"]
  REPLAY["Tracker replay + knob sweep<br/>free"]
  KNOBS["Tracker thresholds"]

  AGENT["Muse Spark agent reviews STT disagreements<br/>keeps a change only if replayed accuracy rises"]
  PROMPTS["Few-shot clips, prompt hints, phrase bank"]

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
  class STT,MATCH,PREQ,STAGES,FUSE,CLIP,GT built
  class STORE,REPLAY,KNOBS,UI building
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
