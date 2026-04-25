"""
Unit tests for the music recommendation system.

Covers the real dict-based pipeline:
  - score_song      : scoring a single song against user preferences
  - recommend_songs : ranking + filtering a candidate list
  - SongRetriever   : vector search over the full catalog
  - Evaluator       : per-round satisfaction and confidence tracking
"""

import pytest
from src.recommender import score_song, recommend_songs, MOOD_ADJACENCY
from src.retriever import SongRetriever
from src.evaluator import Evaluator


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_song(**kwargs):
    defaults = {
        "id": 1, "title": "Test Song", "artist": "Test Artist",
        "genre": "pop", "mood": "happy",
        "energy": 0.8, "valence": 0.8, "danceability": 0.75,
        "acousticness": 0.15, "instrumentalness": 0.02,
        "speechiness": 0.05, "tempo_bpm": 120.0, "liveness": 0.1,
    }
    return {**defaults, **kwargs}


def make_prefs(**kwargs):
    defaults = {
        "genre": "pop", "mood": "happy",
        "energy": 0.8, "valence": 0.8, "danceability": 0.75,
        "acousticness": 0.15, "instrumentalness": 0.02,
        "speechiness": 0.05, "tempo_bpm": 120.0,
    }
    return {**defaults, **kwargs}


# ── score_song ────────────────────────────────────────────────────────────────

class TestScoreSong:
    def test_returns_float_and_string(self):
        score, reason = score_song(make_song(), make_prefs())
        assert isinstance(score, float)
        assert isinstance(reason, str)

    def test_perfect_match_scores_above_10(self):
        score, _ = score_song(make_song(), make_prefs())
        assert score > 10.0

    def test_exact_mood_beats_adjacent_beats_no_match(self):
        prefs = make_prefs(mood="happy")
        # "relaxed" is adjacent to "happy"; "angry" is not
        assert "relaxed" in MOOD_ADJACENCY.get("happy", [])
        score_exact, _    = score_song(make_song(mood="happy"),   prefs)
        score_adjacent, _ = score_song(make_song(mood="relaxed"), prefs)
        score_none, _     = score_song(make_song(mood="angry"),   prefs)
        assert score_exact > score_adjacent > score_none

    def test_adjacent_mood_reason_text(self):
        _, reason = score_song(make_song(mood="relaxed"), make_prefs(mood="happy"))
        assert "close mood match (+1.5)" in reason

    def test_no_mood_match_has_no_mood_reason(self):
        _, reason = score_song(make_song(mood="angry"), make_prefs(mood="happy"))
        assert "mood match" not in reason

    def test_genre_match_increases_score(self):
        score_match, _    = score_song(make_song(genre="pop"),  make_prefs(genre="pop"))
        score_no_match, _ = score_song(make_song(genre="rock"), make_prefs(genre="pop"))
        assert score_match > score_no_match

    def test_closer_energy_scores_higher(self):
        prefs = make_prefs(energy=0.8)
        score_close, _ = score_song(make_song(energy=0.79), prefs)
        score_far, _   = score_song(make_song(energy=0.2),  prefs)
        assert score_close > score_far

    def test_reason_includes_all_components(self):
        _, reason = score_song(make_song(mood="happy", genre="pop"), make_prefs(mood="happy", genre="pop"))
        assert "mood match" in reason
        assert "genre match" in reason
        assert "energy match" in reason

    def test_sparse_prefs_do_not_crash(self):
        score, _ = score_song(make_song(), {"genre": "pop", "mood": "happy"})
        assert isinstance(score, float)

    def test_score_is_never_negative(self):
        # Worst-case: everything mismatched, extreme tempo (40 BPM, range is 60–168)
        song  = make_song(mood="angry", genre="metal", energy=1.0, acousticness=1.0, tempo_bpm=40.0)
        prefs = make_prefs(mood="chill", genre="lofi", energy=0.0, acousticness=0.0, tempo_bpm=168.0)
        score, _ = score_song(song, prefs)
        assert score >= 0.0

    def test_high_tempo_does_not_produce_negative_score(self):
        # Tempo 250 BPM is above the 60–168 normalisation window
        song  = make_song(tempo_bpm=250.0)
        prefs = make_prefs(tempo_bpm=60.0)
        score, _ = score_song(song, prefs)
        assert score >= 0.0


# ── recommend_songs ───────────────────────────────────────────────────────────

class TestRecommendSongs:
    def setup_method(self):
        self.pop  = make_song(id=1, title="Pop Hit",    genre="pop",  mood="happy",   energy=0.8)
        self.lofi = make_song(id=2, title="Lofi Beat",  genre="lofi", mood="chill",   energy=0.3,
                              acousticness=0.9, instrumentalness=0.8)
        self.rock = make_song(id=3, title="Rock Track", genre="rock", mood="intense", energy=0.9,
                              valence=0.4, acousticness=0.05)
        self.songs = [self.pop, self.lofi, self.rock]
        self.prefs = make_prefs(genre="pop", mood="happy", energy=0.8)

    def test_returns_list_of_three_tuples(self):
        results = recommend_songs(self.prefs, self.songs, k=3)
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, tuple) and len(item) == 3
            song, score, reason = item
            assert isinstance(song, dict)
            assert isinstance(score, float)
            assert isinstance(reason, str)

    def test_pop_song_ranks_first_for_pop_prefs(self):
        results = recommend_songs(self.prefs, self.songs, k=3)
        assert results[0][0]["title"] == "Pop Hit"

    def test_results_sorted_descending(self):
        results = recommend_songs(self.prefs, self.songs, k=3)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_respects_k_limit(self):
        assert len(recommend_songs(self.prefs, self.songs, k=2)) == 2

    def test_excludes_liked_song_ids(self):
        prefs = {**self.prefs, "liked_song_ids": [1]}
        ids = [r[0]["id"] for r in recommend_songs(prefs, self.songs, k=3)]
        assert 1 not in ids

    def test_returns_all_songs_when_k_exceeds_catalog(self):
        assert len(recommend_songs(self.prefs, self.songs, k=99)) == 3

    def test_empty_catalog_returns_empty(self):
        assert recommend_songs(self.prefs, [], k=5) == []


# ── SongRetriever ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def retriever():
    return SongRetriever("data/songs_full.csv")


class TestSongRetriever:
    def test_catalog_size_is_large(self, retriever):
        assert retriever.catalog_size > 10_000

    def test_retrieve_returns_list_of_dicts(self, retriever):
        results = retriever.retrieve(make_prefs(), k=10)
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_retrieve_respects_k(self, retriever):
        assert len(retriever.retrieve(make_prefs(), k=7)) <= 7

    def test_each_result_has_similarity_score(self, retriever):
        for r in retriever.retrieve(make_prefs(), k=5):
            assert "_similarity" in r
            assert 0.0 <= r["_similarity"] <= 1.0

    def test_exclude_ids_removes_song(self, retriever):
        first = retriever.retrieve(make_prefs(), k=5)
        excluded_id = first[0]["id"]
        second_ids = [r["id"] for r in retriever.retrieve(make_prefs(), k=5, exclude_ids=[excluded_id])]
        assert excluded_id not in second_ids

    def test_min_popularity_filter_respected(self, retriever):
        results = retriever.retrieve(make_prefs(), k=50, min_popularity=80)
        for r in results:
            assert int(r.get("popularity", 0)) >= 80

    def test_genres_returns_non_empty_collection(self, retriever):
        genres = retriever.genres()
        assert len(genres) > 0

    def test_moods_returns_non_empty_collection(self, retriever):
        moods = retriever.moods()
        assert len(moods) > 0


# ── Evaluator ─────────────────────────────────────────────────────────────────

class TestEvaluator:
    def test_empty_summary(self):
        assert "No rounds" in Evaluator().summary()

    def test_satisfaction_is_liked_over_shown(self):
        ev = Evaluator()
        ev.record_round(1, shown=5, liked=3, confidence=0.7)
        assert ev.satisfaction_scores() == [0.6]

    def test_zero_shown_does_not_crash(self):
        ev = Evaluator()
        ev.record_round(1, shown=0, liked=0, confidence=0.5)
        assert ev.satisfaction_scores() == [0.0]

    def test_is_improving_requires_two_rounds(self):
        ev = Evaluator()
        ev.record_round(1, shown=5, liked=2, confidence=0.6)
        assert ev.is_improving() is False

    def test_is_improving_true(self):
        ev = Evaluator()
        ev.record_round(1, shown=5, liked=2, confidence=0.6)
        ev.record_round(2, shown=5, liked=4, confidence=0.8)
        assert ev.is_improving() is True

    def test_is_improving_false_when_drops(self):
        ev = Evaluator()
        ev.record_round(1, shown=5, liked=5, confidence=0.9)
        ev.record_round(2, shown=5, liked=1, confidence=0.4)
        assert ev.is_improving() is False

    def test_average_confidence(self):
        ev = Evaluator()
        ev.record_round(1, shown=5, liked=3, confidence=0.6)
        ev.record_round(2, shown=5, liked=4, confidence=0.8)
        assert ev.average_confidence() == 0.7

    def test_best_round_is_highest_satisfaction(self):
        ev = Evaluator()
        ev.record_round(1, shown=5, liked=2, confidence=0.5)
        ev.record_round(2, shown=5, liked=5, confidence=0.9)
        ev.record_round(3, shown=5, liked=3, confidence=0.7)
        assert ev.best_round()["round"] == 2

    def test_best_round_empty_returns_empty_dict(self):
        assert Evaluator().best_round() == {}

    def test_summary_contains_key_fields(self):
        ev = Evaluator()
        ev.record_round(1, shown=5, liked=2, confidence=0.6)
        ev.record_round(2, shown=5, liked=4, confidence=0.8)
        s = ev.summary()
        assert "Satisfaction" in s
        assert "confidence" in s.lower()
        assert "Improving" in s

    def test_as_chart_data_has_required_keys(self):
        ev = Evaluator()
        ev.record_round(1, shown=5, liked=3, confidence=0.7)
        data = ev.as_chart_data()
        assert "Round" in data
        assert "Satisfaction %" in data
        assert "Confidence %" in data
