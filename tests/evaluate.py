"""
Test Harness — runs the full recommendation pipeline on predefined scenarios
and prints a structured pass/fail report.

Usage
-----
    python3 tests/evaluate.py

What is tested
--------------
Each scenario verifies 4 things:
  1. RAG retriever returns the expected number of candidates
  2. Feedback agent returns a valid preference schema
  3. Agent confidence score is in the valid range [0.0, 1.0]
  4. Preferences changed after feedback (the agent actually updated something)
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.retriever import SongRetriever
from src.recommender import get_recommendations
from src.feedback_agent import run_feedback_agent, VALID_GENRES, VALID_MOODS
from src.evaluator import Evaluator

# ── Predefined test scenarios ─────────────────────────────────────────────────

SCENARIOS = [
    {
        "name": "High-energy rock fan",
        "prefs": {
            "genre": "rock", "mood": "intense",
            "energy": 0.88, "valence": 0.45, "danceability": 0.65,
            "acousticness": 0.08, "instrumentalness": 0.05,
            "speechiness": 0.06, "tempo_bpm": 148, "liked_song_ids": [],
        },
        "feedback": [
            {"song": {"id": 999, "title": "Storm Runner", "artist": "Voltline",
                      "genre": "rock", "mood": "intense", "energy": 0.91,
                      "valence": 0.48, "danceability": 0.66, "acousticness": 0.10,
                      "instrumentalness": 0.04, "speechiness": 0.06, "tempo_bpm": 152},
             "rating": "like"},
            {"song": {"id": 998, "title": "Library Rain", "artist": "Paper Lanterns",
                      "genre": "lofi", "mood": "chill", "energy": 0.35,
                      "valence": 0.60, "danceability": 0.58, "acousticness": 0.86,
                      "instrumentalness": 0.81, "speechiness": 0.02, "tempo_bpm": 72},
             "rating": "dislike"},
        ],
    },
    {
        "name": "Chill lofi student",
        "prefs": {
            "genre": "lofi", "mood": "chill",
            "energy": 0.35, "valence": 0.55, "danceability": 0.55,
            "acousticness": 0.80, "instrumentalness": 0.75,
            "speechiness": 0.03, "tempo_bpm": 78, "liked_song_ids": [],
        },
        "feedback": [
            {"song": {"id": 997, "title": "Focus Flow", "artist": "LoRoom",
                      "genre": "lofi", "mood": "focused", "energy": 0.40,
                      "valence": 0.59, "danceability": 0.60, "acousticness": 0.78,
                      "instrumentalness": 0.69, "speechiness": 0.03, "tempo_bpm": 80},
             "rating": "like"},
            {"song": {"id": 996, "title": "Drop Zone", "artist": "Circuit9",
                      "genre": "edm", "mood": "energetic", "energy": 0.96,
                      "valence": 0.71, "danceability": 0.94, "acousticness": 0.08,
                      "instrumentalness": 0.62, "speechiness": 0.05, "tempo_bpm": 140},
             "rating": "dislike"},
        ],
    },
    {
        "name": "Happy pop dancer",
        "prefs": {
            "genre": "pop", "mood": "happy",
            "energy": 0.75, "valence": 0.82, "danceability": 0.85,
            "acousticness": 0.15, "instrumentalness": 0.02,
            "speechiness": 0.07, "tempo_bpm": 122, "liked_song_ids": [],
        },
        "feedback": [
            {"song": {"id": 995, "title": "Sunrise City", "artist": "Neon Echo",
                      "genre": "pop", "mood": "happy", "energy": 0.82,
                      "valence": 0.84, "danceability": 0.79, "acousticness": 0.18,
                      "instrumentalness": 0.02, "speechiness": 0.05, "tempo_bpm": 118},
             "rating": "like"},
            {"song": {"id": 994, "title": "Velvet Hours", "artist": "Maison Rouge",
                      "genre": "classical", "mood": "melancholic", "energy": 0.24,
                      "valence": 0.38, "danceability": 0.28, "acousticness": 0.96,
                      "instrumentalness": 0.97, "speechiness": 0.04, "tempo_bpm": 58},
             "rating": "dislike"},
        ],
    },
    {
        "name": "Conflicted user (all likes)",
        "prefs": {
            "genre": "jazz", "mood": "relaxed",
            "energy": 0.40, "valence": 0.65, "danceability": 0.55,
            "acousticness": 0.70, "instrumentalness": 0.40,
            "speechiness": 0.05, "tempo_bpm": 92, "liked_song_ids": [],
        },
        "feedback": [
            {"song": {"id": 993, "title": "Coffee Shop Stories", "artist": "Slow Stereo",
                      "genre": "jazz", "mood": "relaxed", "energy": 0.37,
                      "valence": 0.71, "danceability": 0.54, "acousticness": 0.89,
                      "instrumentalness": 0.43, "speechiness": 0.04, "tempo_bpm": 90},
             "rating": "like"},
            {"song": {"id": 992, "title": "Bali Drift", "artist": "Coastal Current",
                      "genre": "reggae", "mood": "relaxed", "energy": 0.55,
                      "valence": 0.83, "danceability": 0.76, "acousticness": 0.67,
                      "instrumentalness": 0.06, "speechiness": 0.10, "tempo_bpm": 98},
             "rating": "like"},
        ],
    },
    {
        "name": "Adversarial: single dislike only",
        "prefs": {
            "genre": "hip-hop", "mood": "energetic",
            "energy": 0.72, "valence": 0.66, "danceability": 0.85,
            "acousticness": 0.12, "instrumentalness": 0.01,
            "speechiness": 0.28, "tempo_bpm": 94, "liked_song_ids": [],
        },
        "feedback": [
            {"song": {"id": 991, "title": "Gold Chain Feelings", "artist": "Rayven Blvd",
                      "genre": "hip-hop", "mood": "nostalgic", "energy": 0.72,
                      "valence": 0.66, "danceability": 0.85, "acousticness": 0.12,
                      "instrumentalness": 0.00, "speechiness": 0.28, "tempo_bpm": 94},
             "rating": "dislike"},
        ],
    },
]

# ── Required keys in the agent output ────────────────────────────────────────

REQUIRED_PREF_KEYS = [
    "genre", "mood", "energy", "valence", "danceability",
    "acousticness", "instrumentalness", "speechiness", "tempo_bpm",
]

# ── Individual checks ─────────────────────────────────────────────────────────

def check_retriever(retriever, prefs) -> tuple[bool, str]:
    try:
        candidates = retriever.retrieve(prefs, k=20)
        if len(candidates) == 0:
            return False, "No candidates returned"
        return True, f"{len(candidates)} candidates retrieved"
    except Exception as exc:
        return False, f"Exception: {exc}"


def check_schema(result: dict) -> tuple[bool, str]:
    updated = result.get("updated_prefs", {})
    missing = [k for k in REQUIRED_PREF_KEYS if k not in updated]
    if missing:
        return False, f"Missing keys: {missing}"
    return True, "All required keys present"


def check_confidence(result: dict) -> tuple[bool, str]:
    conf = result.get("confidence", -1)
    if not isinstance(conf, (int, float)):
        return False, f"Confidence is not a number: {conf}"
    if not (0.0 <= conf <= 1.0):
        return False, f"Confidence out of range: {conf}"
    return True, f"Confidence = {conf}"


def check_prefs_changed(original: dict, result: dict) -> tuple[bool, str]:
    updated = result.get("updated_prefs", {})
    changed = [k for k in REQUIRED_PREF_KEYS
               if updated.get(k) != original.get(k)]
    if not changed:
        return False, "Preferences unchanged after feedback"
    return True, f"Changed: {', '.join(changed)}"


# ── Runner ────────────────────────────────────────────────────────────────────

def run_harness():
    print("\n" + "=" * 65)
    print("  MUSIC RECOMMENDER — TEST HARNESS")
    print("=" * 65)

    retriever = SongRetriever("data/songs_full.csv")
    evaluator = Evaluator()

    total_checks = 0
    passed_checks = 0
    confidences   = []

    for i, scenario in enumerate(SCENARIOS, 1):
        name  = scenario["name"]
        prefs = scenario["prefs"]
        feedback = scenario["feedback"]

        print(f"\nScenario {i}: {name}")
        print("-" * 50)

        checks = {}

        # Check 1 — retriever
        ok, msg = check_retriever(retriever, prefs)
        checks["Retriever returns candidates"] = (ok, msg)

        # Check 2–4 — agent
        try:
            result = run_feedback_agent(prefs, feedback, round_num=i)
            confidences.append(result["confidence"])

            ok2, msg2 = check_schema(result)
            ok3, msg3 = check_confidence(result)
            ok4, msg4 = check_prefs_changed(prefs, result)

            checks["Agent returns valid schema"]      = (ok2, msg2)
            checks["Confidence in range [0, 1]"]     = (ok3, msg3)
            checks["Preferences updated by agent"]   = (ok4, msg4)

            # Record in evaluator
            liked = sum(1 for f in feedback if f["rating"] == "like")
            evaluator.record_round(i, shown=len(feedback), liked=liked,
                                   confidence=result["confidence"])

        except Exception as exc:
            checks["Agent returns valid schema"]    = (False, f"Exception: {exc}")
            checks["Confidence in range [0, 1]"]   = (False, "Agent failed")
            checks["Preferences updated by agent"]  = (False, "Agent failed")

        # Print results
        for check_name, (ok, msg) in checks.items():
            status = "PASS" if ok else "FAIL"
            icon   = "✅" if ok else "❌"
            print(f"  {icon} [{status}] {check_name}")
            print(f"         → {msg}")
            total_checks += 1
            if ok:
                passed_checks += 1

        # Small delay to avoid hitting rate limits
        if i < len(SCENARIOS):
            time.sleep(2)

    # ── Final report ──────────────────────────────────────────────────────────
    avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Checks passed : {passed_checks}/{total_checks}")
    print(f"  Avg confidence: {avg_conf}")
    print(f"  Evaluator     : {evaluator.summary()}")
    print(f"  Result        : {'PASS ✅' if passed_checks == total_checks else 'PARTIAL ⚠️'}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_harness()
