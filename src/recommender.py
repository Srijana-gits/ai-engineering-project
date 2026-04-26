from typing import List, Dict, Tuple

from src.logger import get_logger, log_retrieval, log_recommendations

logger = get_logger(__name__)


def load_songs(csv_path: str) -> List[Dict]:
    """Reads a songs CSV and returns a list of song dicts with numeric fields cast to float/int."""
    import csv

    numeric_fields = {
        "id": int,
        "energy": float,
        "tempo_bpm": float,
        "valence": float,
        "danceability": float,
        "acousticness": float,
        "liveness": float,
        "instrumentalness": float,
        "speechiness": float,
    }

    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for field, cast in numeric_fields.items():
                row[field] = cast(row[field])
            songs.append(row)

    logger.debug("Loaded %d songs from %s", len(songs), csv_path)
    return songs


MOOD_ADJACENCY = {
    "chill":       ["relaxed", "focused"],
    "relaxed":     ["chill", "happy"],
    "focused":     ["chill", "melancholic"],
    "happy":       ["relaxed", "energetic"],
    "intense":     ["angry", "energetic"],
    "angry":       ["intense"],
    "moody":       ["melancholic", "sad"],
    "melancholic": ["moody", "sad"],
    "sad":         ["melancholic", "moody"],
    "energetic":   ["intense", "happy"],
    "nostalgic":   ["melancholic", "relaxed"],
}


def score_song(song: Dict, user_prefs: Dict) -> Tuple[float, str]:
    """Scores a single song against user preferences and returns (total_score, reasons_string)."""
    score = 0.0
    reasons = []

    # --- Mood (max 3.0) ---
    song_mood = song["mood"]
    user_mood = user_prefs.get("mood", "")
    if song_mood == user_mood:
        score += 3.0
        reasons.append("mood match (+3.0)")
    elif song_mood in MOOD_ADJACENCY.get(user_mood, []):
        score += 1.5
        reasons.append("close mood match (+1.5)")

    # --- Genre (max 2.0) ---
    if song["genre"] == user_prefs.get("genre", ""):
        score += 2.0
        reasons.append("genre match (+2.0)")

    # --- Numeric features: points = max_points * (1 - abs(song_value - user_value)) ---
    numeric_features = [
        ("energy",           "energy",           2.5),
        ("acousticness",     "acousticness",     2.0),
        ("instrumentalness", "instrumentalness", 1.0),
        ("valence",          "valence",          0.8),
        ("danceability",     "danceability",     0.4),
        ("tempo_bpm",        "tempo_bpm",        0.3),
    ]

    TEMPO_MIN, TEMPO_MAX = 60.0, 168.0

    for song_key, pref_key, max_pts in numeric_features:
        if pref_key in user_prefs:
            song_val = song[song_key]
            user_val = user_prefs[pref_key]
            if song_key == "tempo_bpm":
                song_val = (song_val - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN)
                user_val = (user_val - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN)
            pts = max(0.0, max_pts * (1 - abs(song_val - user_val)))
            score += pts
            reasons.append(f"{song_key} match (+{pts:.2f})")

    return round(score, 2), ", ".join(reasons)


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Scores every unheard song, sorts by score descending, and returns the top-k results."""
    liked_ids = user_prefs.get("liked_song_ids", [])

    scored = [
        (song, *score_song(song, user_prefs))
        for song in songs
        if song["id"] not in liked_ids
    ]

    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]


def get_recommendations(
    user_prefs: Dict,
    retriever,
    k: int = 5,
    retrieve_k: int = 100,
    round_num: int = 1,
    min_popularity: int = 0,
) -> List[Tuple[Dict, float, str]]:
    """
    Full RAG pipeline: retrieve candidates → score → return top-k.

    1. The retriever does a cosine-similarity search over 72k songs
       and returns the top `retrieve_k` closest matches.
    2. The existing score_song() logic re-ranks those candidates
       using mood, genre, and numeric feature weights.
    3. The final top-k are returned.

    Parameters
    ----------
    user_prefs  : preference dict (genre, mood, energy, valence, …)
    retriever   : a SongRetriever instance (loaded once at app startup)
    k           : number of final recommendations to return
    retrieve_k  : number of RAG candidates to fetch before re-ranking
    round_num   : current feedback round (for logging)
    """
    exclude_ids     = user_prefs.get("liked_song_ids", [])
    disliked_genres = user_prefs.get("disliked_genres", [])

    # Step 1 — RAG: vector search over full catalog
    candidates = retriever.retrieve(user_prefs, k=retrieve_k, exclude_ids=exclude_ids, min_popularity=min_popularity, disliked_genres=disliked_genres)
    log_retrieval(round_num, user_prefs, len(candidates))
    logger.info("RAG retrieved %d candidates | round=%d", len(candidates), round_num)

    # Step 2 — Re-rank candidates with the scoring function
    results = recommend_songs(user_prefs, candidates, k=k)

    # Step 3 — Log what was recommended
    titles = [r[0]["title"] for r in results]
    log_recommendations(round_num, titles, liked_count=0)
    logger.info("Final recommendations: %s", titles)

    return results
