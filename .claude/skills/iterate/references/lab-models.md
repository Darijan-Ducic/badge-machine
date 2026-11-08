# lab-models: which Meta model reads the lips, and in which modality
- **Readers** (GET /v1/models, 2026-11-07; each passed a text call and a 1-image call): muse-spark-1.1, 1.2, 1.3 (default), 1.2-contributor, 1.3-contributor, muse-wonder-1.0. NOT readers: muse-image-1.0 (image generation, /v1/responses only), sam-3.1, muse-voice-transcribe-1.0. Native `input_video` works only on /v1/responses; chat.completions refuses it.
- **Run** (stored crops only; control = each clip's stored 1.3 samples, $0; `--reuse --no-store` re-scores stored re-reads for $0; `--save` writes Runs-tab records):
  `uv run --env-file .env python tools/exp_models.py --user U --arm MODEL:grids|video [--n 40] [--offset 0] [--k 5] [--reuse] [--save]`
- **Primary metric is content viseme lift:** lift minus the lift of lab-prompt's null text at the read's word count. Raw lift rewards a read for matching its STT's length.
- **Evidence** (Obama, n=15, K=5, Runs control de8a3b6cab6b). Content viseme lift: 1.3 -0.033; 1.1 -0.006 (10/5, p=0.30); 1.2 -0.034 (9/6, p=0.61); 1.3 video -0.014 (10/5, p=0.30). Raw viseme lift ranked 1.2 > 1.1 > 1.3, but that gap was length alone.
- **Lever states:**
  - model: OPEN, no detectable difference. Every arm is at chance on content.
  - native video: PARKED. It bills about 150 input tokens per clip, gives 4.7-word reads and a raw lift of 2/13 (p=0.007).
  - 1.2: valid output 89% (absurd frame numbers), $0.18/turn vs $0.11.
- **Reopen:**
  - When a new Muse version ships, or another model appears in /v1/models.
  - When any lab gets 1.3's content lift above 0 (then compare models on that pipeline).
  - On a second speaker, with ≥40 turns per arm.
  - Video: reopen when the API adds fps control.
