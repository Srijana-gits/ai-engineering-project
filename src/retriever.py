"""
RAG Retriever — vector search over the full song catalog.

Represents every song as an 8-dimensional audio-feature vector and uses
cosine similarity to retrieve the most relevant candidates for a given
user preference profile.  The scoring / ranking step that follows is handled
by recommender.py; this module is only responsible for fast candidate recall.

Usage
-----
    from src.retriever import SongRetriever
    retriever = SongRetriever("data/songs_full.csv")
    candidates = retriever.retrieve(user_prefs, k=100)
"""

import csv
import logging
import os
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Feature configuration ────────────────────────────────────────────────────

# Audio features used to build each song's vector (order matters).
FEATURE_KEYS: List[str] = [
    "energy",
    "valence",
    "danceability",
    "acousticness",
    "liveness",
    "instrumentalness",
    "speechiness",
    "tempo_bpm",          # normalised to [0, 1] before comparison
]

# Tempo range covering the cleaned dataset.
TEMPO_MIN: float = 40.0
TEMPO_MAX: float = 250.0


def _normalise_tempo(value: float) -> float:
    return (value - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN)


# ── Retriever class ───────────────────────────────────────────────────────────

class SongRetriever:
    """
    Loads the full song catalog once and exposes a fast cosine-similarity
    retrieve() method for RAG-style candidate recall.
    """

    def __init__(self, csv_path: str):
        """
        Parameters
        ----------
        csv_path : str
            Path to songs_full.csv (or any CSV matching our schema).
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Song catalog not found: {csv_path}\n"
                "Run  python3 data/prepare_songs.py  first."
            )

        self.songs: List[Dict] = self._load_songs(csv_path)
        self._matrix: np.ndarray = self._build_matrix()   # (N, 8)  float32
        logger.info("SongRetriever loaded %d songs from %s", len(self.songs), csv_path)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_songs(self, csv_path: str) -> List[Dict]:
        """Reads the CSV and casts numeric fields."""
        numeric_fields = {
            "id":               int,
            "popularity":       int,
            "energy":           float,
            "tempo_bpm":        float,
            "valence":          float,
            "danceability":     float,
            "acousticness":     float,
            "liveness":         float,
            "instrumentalness": float,
            "speechiness":      float,
        }
        songs = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    for field, cast in numeric_fields.items():
                        row[field] = cast(row[field])
                    songs.append(row)
                except (ValueError, KeyError) as exc:
                    logger.warning("Skipping malformed row: %s", exc)
        return songs

    def _build_matrix(self) -> np.ndarray:
        """
        Builds an (N, 8) float32 matrix where each row is a song's
        normalised audio-feature vector.
        """
        rows = []
        for song in self.songs:
            vec = [
                song["energy"],
                song["valence"],
                song["danceability"],
                song["acousticness"],
                song["liveness"],
                song["instrumentalness"],
                song["speechiness"],
                _normalise_tempo(song["tempo_bpm"]),
            ]
            rows.append(vec)

        matrix = np.array(rows, dtype=np.float32)

        # L2-normalise every row so dot-product == cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)   # avoid division by zero
        return matrix / norms

    def _user_vector(self, user_prefs: Dict) -> np.ndarray:
        """
        Builds a normalised query vector from the user's preference dict.
        Missing keys fall back to neutral midpoint (0.5).
        """
        vec = np.array([
            float(user_prefs.get("energy",           0.5)),
            float(user_prefs.get("valence",           0.5)),
            float(user_prefs.get("danceability",      0.5)),
            float(user_prefs.get("acousticness",      0.5)),
            float(user_prefs.get("liveness",          0.5)),
            float(user_prefs.get("instrumentalness",  0.5)),
            float(user_prefs.get("speechiness",       0.5)),
            _normalise_tempo(float(user_prefs.get("tempo_bpm", 120.0))),
        ], dtype=np.float32)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        user_prefs: Dict,
        k: int = 100,
        exclude_ids: Optional[List[int]] = None,
        min_popularity: int = 0,
    ) -> List[Dict]:
        """
        Returns the top-k songs most similar to user_prefs by cosine similarity.

        Parameters
        ----------
        user_prefs   : dict with audio-feature keys (see FEATURE_KEYS)
        k            : number of candidates to return
        exclude_ids  : song IDs to exclude (already heard / rated)

        Returns
        -------
        List of song dicts sorted by similarity descending.
        """
        exclude_set = set(exclude_ids or [])

        query = self._user_vector(user_prefs)                   # (8,)
        scores = self._matrix @ query                            # (N,)  cosine sim

        # Sort indices by similarity descending
        ranked_indices = np.argsort(scores)[::-1]

        candidates = []
        for idx in ranked_indices:
            if len(candidates) >= k:
                break
            song = self.songs[idx]
            if song["id"] not in exclude_set and song.get("popularity", 0) >= min_popularity:
                candidates.append({**song, "_similarity": float(scores[idx])})

        logger.debug(
            "retrieve(): top similarity=%.3f  bottom=%.3f  returned=%d",
            candidates[0]["_similarity"] if candidates else 0,
            candidates[-1]["_similarity"] if candidates else 0,
            len(candidates),
        )
        return candidates

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def catalog_size(self) -> int:
        return len(self.songs)

    def genres(self) -> List[str]:
        return sorted(set(s["genre"] for s in self.songs))

    def moods(self) -> List[str]:
        return sorted(set(s["mood"] for s in self.songs))
