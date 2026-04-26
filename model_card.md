# Model Card: Music Recommender with AI Feedback Loop

## What this system does

This system recommends songs based on a user's stated preferences: genre, mood, energy, valence, tempo and improves those recommendations round-by-round using an AI feedback agent. After you rate songs (like/dislike), Google Gemini 2.0 Flash reads your feedback, adjusts your preference profile, and the next round reflects those changes.

Built as a learning project to explore RAG pipelines, LLM tool use, and evaluation loops. Not for production use.

---

## Base project

The starting point was a simple 20-song catalog with a single scoring function (`score_song`) and a command-line interface (`src/main.py`). From that foundation, the project was extended to add:
- A 72k-song catalog from a public Spotify dataset
- Vector similarity retrieval (`src/retriever.py`) to replace brute-force scoring
- A Gemini LLM feedback agent (`src/feedback_agent.py`) to update preferences between rounds
- A Streamlit web UI (`app.py`) replacing the CLI
- Structured logging (`src/logger.py`) and an evaluation layer (`src/evaluator.py`)
- A full unit test suite (37 tests) and an integration harness

---

## How it works

**Retrieval** — User preferences are encoded as a feature vector and compared against 72k songs using cosine similarity. The top 100 candidates are returned.

**Ranking** — Each candidate is scored: mood (max 3.0 pts), genre (max 2.0 pts), and six numeric audio features weighted by relevance. Top 5 are shown.

**AI Feedback** — The LLM runs three tool calls in sequence: `analyze_feedback` → `update_preferences` → `set_confidence`. It edits the preference vector directly based on what you liked and disliked.

**Evaluation** — Satisfaction % (liked / shown) and agent confidence are tracked per round. An "Improving" flag turns positive when the trend is upward.

---

## Data

The catalog is `data/songs_full.csv` — 72,249 songs derived from a public Spotify dataset. Each song has audio features (energy, valence, tempo, danceability, acousticness, instrumentalness, speechiness) plus mood and genre labels assigned during preprocessing.

The dataset reflects what was streamed heavily on Spotify. Less-streamed genres and non-English music are underrepresented relative to their actual presence in the world.

---

## Limitations and biases

**Mood dominates scoring.** A wrong mood match costs up to 3.0 pts — the largest single weight. A song that matches on every other feature but has the wrong mood will rank below a mediocre song with the right mood. This is intentional (mood reflects why you're listening right now), but it means the system is sensitive to mood misclassification in the catalog.

**Popularity bias in the catalog.** `songs_full.csv` comes from Spotify streaming data. Artists and genres with large streaming audiences are overrepresented. Niche or regional music is harder to surface even if it would be a better match.

**Stated preferences aren't always accurate.** The system only knows what you tell it. Users who can't accurately describe their taste in numeric terms (energy=0.7, valence=0.5) get worse results than users who know exactly what they want.

**The agent updates regardless of confidence.** Even at 50% confidence, the LLM edits the preference vector. Low-confidence updates can push the profile in the wrong direction before enough feedback exists to correct it.

**No cross-session memory.** Liked song IDs and session history reset on page reload. Every new session starts from scratch.

**Tempo has minimal influence.** Max 0.3 pts. If BPM matters to you, this system mostly ignores it.

---

## Could this system be misused?

A music recommender has a low misuse surface compared to most AI systems. The realistic risks are:

**Preference profiling.** The system learns what you like and adjusts to match it. If deployed with user accounts and persistent storage, that behavioral data could be used to infer personality traits or emotional states beyond music taste. In this project there are no accounts and no persistent storage, so this risk doesn't apply — but it would matter in a real deployment.

**API key exposure.** The Gemini API key is stored in `.env`. If that file were committed to a public repository, the key could be used to run LLM calls at the account holder's expense. Mitigation: `.env` is excluded in `.gitignore` and `.env.example` is provided so users know the expected format without committing real credentials.

**Catalog manipulation.** If the song catalog were user-editable, someone could inject entries with extreme feature values to always rank first. The current system reads from a static file, so this is not a real risk here.

The system doesn't make consequential decisions, handle sensitive personal data, or interact with external services beyond Google's Gemini inference API. The responsible use concerns are real but narrow in scope for a system of this kind.

---

## What surprised me while testing reliability

**The negative score bug.** I wrote a test (`test_score_is_never_negative`) that fed a song with tempo=250 BPM into the scoring function and checked that the score stayed ≥ 0. It failed. The scoring formula was producing negative contributions for songs with tempo outside the normalized range [60, 168] BPM and the catalog has songs well outside that range. The system had been silently penalizing songs for having "too high" a tempo, and I never would have caught it from the output alone because final scores still looked reasonable. The fix was one line: `pts = max(0.0, ...)`. What surprised me was that the test caught the bug before I even understood the root cause.

**The LLM feedback being more coherent than expected.** I assumed the agent would occasionally produce invalid JSON, hallucinate preference keys, or drift randomly over multiple rounds. Instead, the tool calls returned valid structured output consistently, the reasoning explanations were clear and matched the ratings given, and the preference updates moved in a direction that made sense. Satisfaction went from 60% to 80% between rounds 1 and 2 in live testing.

---

## Collaboration with AI during this project

I used Claude (claude-sonnet-4-6) throughout, for code review, debugging, and writing parts of the system.

**One instance where it was genuinely helpful:**
When I asked Claude to review the test file, it found that the existing tests were testing stubs. `Recommender.recommend()` returned `self.songs[:k]` (unsorted, no filtering), and `explain_recommendation()` returned the string `"Explanation placeholder"`. The tests passed because the stubs never raised errors — they weren't testing the actual logic at all. Claude rewrote the test suite to target the real functions (`score_song`, `recommend_songs`, `SongRetriever`, `Evaluator`), and those tests immediately surfaced the negative score bug described above.

**One instance where its suggestion was flawed:**
Early in the project, Claude suggested keeping the `Recommender` class as a clean wrapper around the scoring logic. I followed that recommendation. The class ended up being dead code, none of its methods were called by the actual pipeline, and two of them (`recommend()` and `explain_recommendation()`) were unimplemented stubs that returned hardcoded placeholders. Removing it later required rewriting the entire test suite from scratch because the old tests were built around that class. The suggestion added structural complexity without adding value, and I should have questioned whether a wrapper class was necessary before accepting it.

---

## Evaluation summary

**Unit tests:** 37 tests across 4 classes — `TestScoreSong`, `TestRecommendSongs`, `TestSongRetriever`, `TestEvaluator`. All pass in under 0.4 seconds.

**Integration harness:** `tests/evaluate.py` runs 5 scenarios through the full pipeline: retrieval, agent, evaluator, and validates schema correctness, confidence bounds, and that preference drift direction matches the feedback given.

**Live session result:** Starting from a pop/happy/high-energy profile, satisfaction improved from 60% to 80% across two rounds. Agent average confidence: 0.85. Evaluator result: Improving: Yes.

---

## What I'd change

Give the agent a confidence threshold below which it skips updating preferences, 50% confidence is not enough signal to be adjusting sliders. Add session persistence (SQLite) so liked songs and history survive a page reload. Surface the agent's reasoning directly in the UI so users understand why their recommendations changed, not just that they did. Replace the hand-tuned scoring weights with a learned ranker once enough rating history exists to train on.
