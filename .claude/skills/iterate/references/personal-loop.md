# Personal loop — fitting the system to one user

What is learned per user: scorer weights (staged), the auto-send threshold, the phrase bank and its frequencies (the
`prior` feature), and per-phrase readiness for lip-only mode. Planned: few-shot examples of the user's own clips,
camera framing calibration, and confirm/edit labels from video-only mode. A personal model fits the CURRENT system
to one user's clips; it never changes the system, the labels, or the raw outputs, and it can always be re-fit from
stored raw outputs.

## Read first: what the code does today

- The runtime re-fits each user's scorer from all their earlier labelled clips on every request (POST /preheat/clip,
  POST /vtt, each dictation turn, GET /stats, GET /clips…), and nothing persists the model. Stages unlock by count at
  10 / 30 / 60 labelled clips, and the threshold is re-tuned by 5-fold cross-validation at precision 0.9. See
  [instruments.md](instruments.md) → What runs without you.
- That automatic re-fit is what the user ruled out. Replacing it with a personal version you adopt is missing
  instrument (f).
- Until (f) lands, the personal loop is **observe, judge, record**: you diagnose and judge, write each decision to
  [history.md](history.md), and don't hand-edit runtime weights.

## When to iterate on a user

- The user crosses a stage boundary (default plan: 10 / 30 / 60 labelled clips). Check that the newly fitted stage
  beats the previous one on the user's newest clips before letting it stand.
- New labelled clips have arrived since the user's last personal version, or since your last check.
- The user corrected reads (confirm/edit labels), or the Stats page shows recent accuracy dropping.
- A new system version went live: every user's personal layers are re-fit under it. Check each active user.

## One personal iteration

1. **Look.** `tools/diagnose.py recipe --user U` (stage, the weights this stage fits, the threshold, pool recall);
   `failures --user U`; the Stats page learning curve (predict-then-train) ([tool-diagnose.md](tool-diagnose.md)).
2. **Judge** the re-fit on the user's newest clips that it wasn't fitted on (the held-out part of that user's stream):
   word and viseme accuracy, auto-send precision and coverage.
3. **Decide:** keep the previous personal version, adopt the re-fit, or hold the stage back (e.g. stay on the starting
   weights when the fitted stage is worse at this clip count).
4. **Per-user levers** you may change: stage timing, the threshold target, bank curation (drop phrases the user never
   says, add frequent ones), and the readiness rule. Few-shot examples and camera calibration come once they're built.
5. **Record:** a personal version = (system version, user, `trained_on`, stage, threshold, author = this session),
   once (f) lands; until then, a history.md entry.
6. **Readiness:** lip-only mode unlocks a phrase by the readiness rule (readiness.py: N_MIN 5, WINDOW 5, ACC_MIN 0.8);
   you check it. Natural one-off sentences never become ready; that is expected.

## Personal or system?

- The same failure across several users is a system lever: take it to [system-loop.md](system-loop.md).
- A failure specific to one user (their vocabulary, their camera, their mouth) stays personal.
