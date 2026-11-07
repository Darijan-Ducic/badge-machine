# Personal loop — fitting the system to one user

What is learned per user: scorer weights (staged), the auto-send threshold, and the phrase bank with its frequencies
(the `prior` feature); per-phrase readiness follows from them. Planned: few-shot examples of the user's own clips,
camera framing calibration, and confirm/edit labels from video-only mode. A personal model fits the CURRENT system to
one user's clips. It never changes the system, the labels, or the raw outputs, and it can always be re-fit from stored
raw outputs. Run records for this loop are tagged `"loop": "personal"`; see also lab-personal.md once it lands.

## Read first: what the code does today

- The runtime re-fits each user's scorer from all their earlier labelled clips on every request (POST /preheat/clip,
  POST /vtt, each dictation turn, GET /stats, GET /clips…), and nothing persists the model. Stages unlock by count at
  10 / 30 / 60 labelled clips, and the threshold is re-tuned by 5-fold cross-validation at precision 0.9. See
  [instruments.md](instruments.md) → What runs without you.
- That automatic re-fit is what the user ruled out. Replacing it with a personal version you adopt is missing
  instrument (f).
- The stage plan, threshold target, readiness rule and bank curation are **global** settings today (scorer.py:27, 81;
  readiness.py:15-17; `versions` config). Changing one changes every user, so it is a system change: take it to
  [system-loop.md](system-loop.md) with evidence from several users. Per-user settings arrive with (f).
- Until (f) lands, the personal loop is **observe, judge, recommend, record**: you diagnose and judge, write each
  recommendation to [history.md](history.md), and don't hand-edit runtime weights or global settings for one user.

## When to look at a user

- The user crosses a stage boundary (default plan: 10 / 30 / 60 labelled clips). Check whether the newly fitted stage
  beats the previous one.
- New labelled clips have arrived since your last check.
- The user corrected reads (confirm/edit labels), or the Stats page curve drops.
- Once the runtime reads `live` (missing instrument (e)): a new system version went live, so every user's personal
  layers re-fit under it. Check each active user.

## One personal check

1. **Look.** `tools/diagnose.py recipe --user U` (stage, the weights this stage fits, the threshold, pool recall);
   `failures --user U`; the Stats page curve (predict-then-train) ([tool-diagnose.md](tool-diagnose.md)).
2. **Judge** on the user's **held-out** split, the middle 20% of their stream: word and viseme accuracy, auto-send
   precision and coverage. NEVER judge on the newest 20%, because that is the test split.
3. **Decide and recommend:** keep the previous stage, take the re-fit, or hold the stage back (e.g. stay on the
   starting weights when the fitted stage is worse at this clip count). Until (f) lands there is no per-user switch,
   so write the recommendation and its evidence to history.md and tell the user.
4. **Record:** once (f) lands, a personal version = (system version, user, `trained_on`, stage, threshold,
   `author="agent"`, the session id in the reason). Until then, a history.md entry.
5. **Readiness:** lip-only mode unlocks a phrase by the readiness rule (readiness.py: N_MIN 5, WINDOW 5, ACC_MIN 0.8);
   check it free at `GET /readiness?user=U`. Natural one-off sentences never become ready; that is expected.

## Personal or system?

- The same failure across several users is a system lever: take it to [system-loop.md](system-loop.md).
- A failure specific to one user (their vocabulary, their camera, their mouth) stays personal. Today you can only
  record it and recommend; acting on it per user needs (f).
