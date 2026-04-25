"""
Prepares the Spotify Kaggle dataset for use in the recommender system.

Input:  data/dataset.csv   (~114k songs from Kaggle)
Output: data/songs_full.csv (cleaned, schema-matched, mood-labeled)

Run: python3 data/prepare_songs.py
"""

import csv
import os

INPUT  = os.path.join(os.path.dirname(__file__), "dataset.csv")
OUTPUT = os.path.join(os.path.dirname(__file__), "songs_full.csv")

FIELDNAMES = [
    "id", "title", "artist", "genre", "mood", "popularity",
    "energy", "tempo_bpm", "valence", "danceability",
    "acousticness", "liveness", "instrumentalness", "speechiness",
]


def derive_mood(energy: float, valence: float) -> str:
    """
    Maps (energy, valence) to a mood label using Russell's circumplex model.

        High valence + high energy  → energetic / happy
        High valence + low energy   → chill / relaxed
        Low valence  + high energy  → angry / intense
        Low valence  + low energy   → sad / melancholic
    """
    if energy >= 0.7 and valence >= 0.6:
        return "energetic"
    elif energy >= 0.7 and valence >= 0.35:
        return "happy"
    elif energy >= 0.7:
        return "angry"
    elif energy >= 0.5 and valence >= 0.6:
        return "happy"
    elif energy >= 0.5 and valence >= 0.35:
        return "focused"
    elif energy >= 0.5:
        return "moody"
    elif energy >= 0.3 and valence >= 0.6:
        return "chill"
    elif energy >= 0.3 and valence >= 0.35:
        return "relaxed"
    elif valence < 0.2:
        return "sad"
    else:
        return "melancholic"


def prepare():
    songs = []
    seen_titles: set = set()
    skipped = 0

    with open(INPUT, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                energy           = float(row["energy"])
                valence          = float(row["valence"])
                tempo            = float(row["tempo"])
                danceability     = float(row["danceability"])
                acousticness     = float(row["acousticness"])
                liveness         = float(row["liveness"])
                instrumentalness = float(row["instrumentalness"])
                speechiness      = float(row["speechiness"])
                popularity       = int(float(row["popularity"]))
            except (ValueError, KeyError):
                skipped += 1
                continue

            # Drop rows with out-of-range tempo
            if not (40.0 <= tempo <= 250.0):
                skipped += 1
                continue

            # Drop rows where unit-interval features are out of [0, 1]
            unit_features = [energy, valence, danceability, acousticness,
                             liveness, instrumentalness, speechiness]
            if not all(0.0 <= v <= 1.0 for v in unit_features):
                skipped += 1
                continue

            title = row["track_name"].strip()
            if not title:
                skipped += 1
                continue

            # Skip duplicate titles
            if title.lower() in seen_titles:
                skipped += 1
                continue
            seen_titles.add(title.lower())

            # Use only the first artist when multiple are listed
            artist = row["artists"].split(";")[0].strip()
            genre  = row["track_genre"].strip()
            mood   = derive_mood(energy, valence)

            songs.append({
                "id":              len(songs) + 1,
                "title":           title,
                "artist":          artist,
                "genre":           genre,
                "mood":            mood,
                "popularity":      popularity,
                "energy":          round(energy,           4),
                "tempo_bpm":       round(tempo,            3),
                "valence":         round(valence,          4),
                "danceability":    round(danceability,     4),
                "acousticness":    round(acousticness,     4),
                "liveness":        round(liveness,         4),
                "instrumentalness":round(instrumentalness, 6),
                "speechiness":     round(speechiness,      4),
            })

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(songs)

    # ── Summary ──────────────────────────────────────────────────────────────
    genres = sorted(set(s["genre"] for s in songs))
    moods  = sorted(set(s["mood"]  for s in songs))

    print(f"Done.")
    print(f"  Songs saved : {len(songs):,}")
    print(f"  Rows skipped: {skipped:,}")
    print(f"  Genres ({len(genres)}): {', '.join(genres)}")
    print(f"  Moods  ({len(moods)}):  {', '.join(moods)}")
    print(f"  Output: {OUTPUT}")


if __name__ == "__main__":
    prepare()
