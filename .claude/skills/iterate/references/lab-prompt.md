# lab-prompt: Muse open call prompt, sampling and context (`tools/exp_prompt.py`)
Set `obama-r1` = the first 40 processed turns of practice:obama-weekly:obama (clip ids in each run record). Paid arms run at K=3 against the same turns' stored samples[:3]. Re-reads are stored as `exp:lab-prompt-<arm>:<user>`; runs carry author `agent:lab-prompt` and loop `framework`.

| arm (n=40) | lift V / W | content lift V / W | Δ lift vs base-k3, V and W [95% CI] · sign p | $ | run |
|---|---|---|---|---|---|
| base-k3 (stored) | +0.057 / +0.084 | -0.013 / -0.013 | — | 0 | 8a2040dd7f13 |
| prior-only: middle frame held still, p3 prompt | 0 / 0 (120/120 samples: no speech) | 0 / 0 | -0.057, -0.084 · p<0.001 | 1.83 | 6e122c756acd |
| ctx-history8: STT of the 8 earlier turns | +0.132 / +0.178 | +0.039 / +0.068 | +0.075 [+0.040,+0.114], +0.094 [+0.041,+0.154] · p 0.006 / 0.001 | 2.92 | 91e8aef2f619 |

- LIFT IS LENGTH-GAMEABLE. Generic speech cut to Muse's read word count gets +0.069 / +0.097, more than Muse. Cut to the STT's own length it gets +0.26 / +0.40. ALWAYS report content lift = lift(read) - lift(`NULL_WORDS` cut to the read's word count). Muse alone has content lift of about 0: its reads carry nothing turn-specific beyond a length it takes from the clip's duration. Raw accuracy is worse: a generic sentence scores viseme 0.244, the same as Muse.
- The prior-only control cannot measure a guessing prior, because Muse refuses to read a still mouth. Use the length null as the prior baseline.
- context (history, n=8): **inconclusive, 1 speaker**. It holds on Obama, but a framework change needs ≥ 2 speakers. Next: the same arm on practice:fauci-messages:fauci once it has ≥ 15 turns. No leak: the no-leak tests in tests/test_exp_prompt.py pass, and 0/40 contexts share a 5-gram with their turn. The gain is a language prior from earlier turns of the same address. That is valid for dictation, but it is not lip reading.
- K: **parked**. K=3 equals K=5: picks are identical on 40/40 turns, and K=5 agreement is 0.21. Reopen if sample agreement rises above 0.5 (for example, at a lower temperature).
- Untested (paused for the demo): temperature 0.3 and 0.7, effort medium, prompt wording, session-scope context. levers.md also assigns sampling to lab-decode.
- Re-run with `uv run --env-file .env python tools/exp_prompt.py run --set obama-r1 --arm ctx-history8 --k 3 --context history --context-n 8 --lever "language context" --hypothesis "..."`. Stored re-reads are reused, so this costs $0. Also `... free --set obama-r1`, and `... freeze --user U --set NAME --max 40` for a new set.
