import re
import streamlit as st

from src.retriever import SongRetriever
from src.recommender import get_recommendations
from src.feedback_agent import run_feedback_agent
from src.evaluator import Evaluator

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="🎵 Music Recommender", page_icon="🎵", layout="wide")

st.markdown("""
<style>
  /* ── Global background ── */
  .stApp { background-color: #0d0b14; }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] { background-color: #110f1a; }
  section[data-testid="stSidebar"] * { color: #d1c4e9 !important; }

  /* ── Card hover glow ── */
  .vibe-card {
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
  }
  .vibe-card:hover {
    box-shadow: 0 0 18px 2px rgba(139, 92, 246, 0.35);
    border-color: #7c3aed !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Load retriever once (cached across reruns) ────────────────────────────────
@st.cache_resource
def load_retriever():
    return SongRetriever("data/songs_full.csv")

retriever = load_retriever()

# ── Session state defaults ────────────────────────────────────────────────────
if "round_num"     not in st.session_state:
    st.session_state.round_num     = 1
if "feedback"      not in st.session_state:
    st.session_state.feedback      = {}
if "agent_result"  not in st.session_state:
    st.session_state.agent_result  = None
if "current_prefs" not in st.session_state:
    st.session_state.current_prefs = None
if "evaluator"     not in st.session_state:
    st.session_state.evaluator     = Evaluator()

# ── UI helpers ────────────────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    """Purple tones for high/mid match, grey for low — fits the midnight theme."""
    if score >= 10: return "#7c3aed"
    if score >= 7:  return "#a855f7"
    return "#6b7280"


def _vibe_chips(energy: float, valence: float, danceability: float) -> str:
    """
    Three plain-English pill badges describing the song's feel.
    Much easier to scan than thin progress bars.
    """
    def chip(text: str, bg: str, fg: str = "#fff") -> str:
        return (
            f'<span style="background:{bg};color:{fg};padding:5px 11px;'
            f'border-radius:20px;font-size:12px;white-space:nowrap;">{text}</span>'
        )

    if energy >= 0.66:
        e_chip = chip("🔥 Intense",   "#7f1d1d", "#fca5a5")
    elif energy >= 0.35:
        e_chip = chip("⚡ Moderate",  "#374151", "#d1d5db")
    else:
        e_chip = chip("🌙 Chill",     "#1e3a5f", "#93c5fd")

    if valence >= 0.66:
        v_chip = chip("😊 Uplifting",   "#14290c", "#86efac")
    elif valence >= 0.35:
        v_chip = chip("😐 Neutral",     "#374151", "#d1d5db")
    else:
        v_chip = chip("😔 Melancholic", "#1f2937", "#9ca3af")

    if danceability >= 0.66:
        d_chip = chip("💃 Dance floor", "#4a1d96", "#c4b5fd")
    elif danceability >= 0.35:
        d_chip = chip("🚶 Groovy",      "#312e81", "#a5b4fc")
    else:
        d_chip = chip("🪑 Laid-back",   "#1f2937", "#9ca3af")

    return (
        f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin:10px 0;">'
        f'  {e_chip}{v_chip}{d_chip}'
        f'</div>'
    )


def _match_badges(explanation: str) -> str:
    """
    Parse the score explanation string and render each component as a
    colour-coded pill badge.

    Colour logic:
      - mood match      → solid green  (exact, highest weight)
      - close mood      → amber        (adjacent mood, partial credit)
      - genre match     → purple       (categorical exact match)
      - numeric ≥ 1.0   → solid blue   (strong feature alignment)
      - numeric < 1.0   → dim blue     (weak alignment, still shown)
      - numeric < 0.2   → skipped      (too small to be meaningful)
    """
    badges = []
    for part in explanation.split(", "):
        m = re.match(r'(.+?)\s+match\s+\(\+([0-9.]+)\)', part.strip())
        if not m:
            continue
        label, pts = m.group(1).strip(), float(m.group(2))

        if label == "mood":
            bg, fg = "#16a34a", "#fff"
        elif label == "close mood":
            bg, fg = "#d97706", "#fff"
        elif label == "genre":
            bg, fg = "#7c3aed", "#fff"
        else:
            if pts < 0.2:
                continue
            bg = "#0369a1" if pts >= 1.0 else "#1e3a5f"
            fg = "#fff"    if pts >= 1.0 else "#93c5fd"

        display = label.replace("_bpm", " BPM").replace("_", " ")
        badges.append(
            f'<span style="background:{bg};color:{fg};padding:3px 9px;'
            f'border-radius:12px;font-size:11px;white-space:nowrap;">'
            f'{display} +{pts}</span>'
        )
    return " ".join(badges)


def _song_card(rank: int, song: dict, score: float, explanation: str,
               current_rating: str | None) -> str:
    """
    Build the HTML for one song card.

    Layout (top to bottom):
      • Title + artist + YouTube link + current rating indicator  |  Score badge
      • Genre / Mood / Popularity chips
      • Energy / Valence / Danceability vibe chips (plain-English labels)
      • Match reason badges
    """
    sc  = _score_color(score)
    yt  = (
        "https://www.youtube.com/results?search_query="
        + f"{song['title']} {song['artist']}".replace(" ", "+")
    )

    rated = ""
    if current_rating == "like":
        rated = '<span style="margin-left:8px;">✅</span>'
    elif current_rating == "dislike":
        rated = '<span style="margin-left:8px;">❌</span>'

    vibe   = _vibe_chips(float(song["energy"]), float(song["valence"]), float(song["danceability"]))
    badges = _match_badges(explanation)

    genre_chip = (f'<span style="background:#1e3a5f;color:#93c5fd;padding:2px 8px;'
                  f'border-radius:10px;font-size:11px;">{song["genre"]}</span>')
    mood_chip  = (f'<span style="background:#2d1b69;color:#c4b5fd;padding:2px 8px;'
                  f'border-radius:10px;font-size:11px;">{song["mood"]}</span>')
    pop_chip   = (f'<span style="background:#14290c;color:#6ee7b7;padding:2px 8px;'
                  f'border-radius:10px;font-size:11px;">⭐ {song.get("popularity", "?")} / 100</span>')

    return (
        f'<div class="vibe-card" style="border:1px solid #2d1f4e;border-radius:12px;padding:16px;'
        f'margin-bottom:4px;background:#13111c;">'
        f'  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">'
        f'    <div>'
        f'      <span style="font-size:16px;font-weight:700;">#{rank}&nbsp;{song["title"]}</span>'
        f'      <span style="color:#9ca3af;">&nbsp;—&nbsp;{song["artist"]}</span>'
        f'      <a href="{yt}" target="_blank"'
        f'         style="color:#60a5fa;margin-left:8px;font-size:13px;text-decoration:none;">'
        f'        🔍 YouTube</a>'
        f'      {rated}'
        f'    </div>'
        f'    <div style="background:{sc};color:#fff;padding:4px 12px;border-radius:20px;'
        f'         font-size:13px;font-weight:600;white-space:nowrap;">{score} pts</div>'
        f'  </div>'
        f'  <div style="display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap;">'
        f'    {genre_chip}&nbsp;{mood_chip}&nbsp;{pop_chip}'
        f'  </div>'
        f'  {vibe}'
        f'  <div style="display:flex;flex-wrap:wrap;gap:4px;">{badges}</div>'
        f'</div>'
    )


# ── Sidebar — initial preferences ─────────────────────────────────────────────
st.sidebar.header("Your Preferences")
st.sidebar.caption("Set your starting taste. The AI will refine it from your ratings.")

GENRES = sorted([
    "acoustic", "alt-rock", "ambient", "blues", "classical", "country",
    "dance", "edm", "electronic", "folk", "funk", "hip-hop", "house",
    "indie", "indie-pop", "jazz", "latin", "metal", "pop", "punk",
    "r-n-b", "reggae", "rock", "soul", "techno", "trip-hop",
])
MOODS = retriever.moods()

# Grouped into three collapsible sections so the sidebar isn't one long wall of sliders.
with st.sidebar.expander("🎵 Vibe", expanded=True):
    genre     = st.selectbox("Genre", GENRES, index=GENRES.index("pop"))
    mood      = st.selectbox("Mood",  MOODS,  index=MOODS.index("happy"))
    tempo_bpm = st.slider("Tempo (BPM)", 40.0, 250.0, 120.0, 1.0)

with st.sidebar.expander("🎛️ Audio Features", expanded=False):
    energy           = st.slider("Energy",           0.0, 1.0, 0.5,  0.01)
    valence          = st.slider("Valence",          0.0, 1.0, 0.5,  0.01)
    danceability     = st.slider("Danceability",     0.0, 1.0, 0.5,  0.01)
    acousticness     = st.slider("Acousticness",     0.0, 1.0, 0.3,  0.01)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.1,  0.01)
    speechiness      = st.slider("Speechiness",      0.0, 1.0, 0.05, 0.01)

with st.sidebar.expander("⚙️ Results", expanded=False):
    k              = st.slider("Recommendations",                          1,   10,  5)
    popularity_min = st.slider("Min Popularity (0 = all, 60 = well-known)", 0,  100, 50, 5)

sidebar_prefs = {
    "genre":            genre,
    "mood":             mood,
    "energy":           energy,
    "valence":          valence,
    "danceability":     danceability,
    "acousticness":     acousticness,
    "instrumentalness": instrumentalness,
    "speechiness":      speechiness,
    "tempo_bpm":        tempo_bpm,
    "liked_song_ids":   [],
}

if st.session_state.current_prefs is None:
    st.session_state.current_prefs = sidebar_prefs

st.sidebar.divider()
if st.sidebar.button("🔄 Reset Session"):
    st.session_state.round_num     = 1
    st.session_state.feedback      = {}
    st.session_state.agent_result  = None
    st.session_state.current_prefs = sidebar_prefs
    st.session_state.evaluator     = Evaluator()
    st.rerun()

# Show AI-updated profile snapshot when the agent has run at least once
if st.session_state.round_num > 1 and st.session_state.current_prefs:
    st.sidebar.divider()
    st.sidebar.caption("**AI-updated profile:**")
    p = st.session_state.current_prefs
    st.sidebar.caption(
        f"Genre: {p.get('genre')}  |  Mood: {p.get('mood')}\n"
        f"Energy: {p.get('energy')}  |  Valence: {p.get('valence')}"
    )

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:24px 0 8px 0;border-bottom:1px solid #2d1f4e;margin-bottom:20px;">
  <div style="display:flex;align-items:center;gap:16px;">
    <div style="font-size:42px;line-height:1;">🎵</div>
    <div>
      <div style="font-size:28px;font-weight:800;letter-spacing:-0.5px;
                  background:linear-gradient(90deg,#a78bfa,#ec4899);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  background-clip:text;">
        Music Recommender
      </div>
      <div style="font-size:13px;color:#6b7280;margin-top:2px;">
        RAG over {retriever.catalog_size:,} songs · powered by an AI feedback agent
      </div>
    </div>
  </div>
  <!-- Round counter pill -->
  <div style="background:#1e1b4b;border:1px solid #4338ca;color:#a5b4fc;
              padding:6px 16px;border-radius:20px;font-size:13px;font-weight:600;">
    Round {st.session_state.round_num}
  </div>
</div>
""", unsafe_allow_html=True)

# ── Agent insight panel ────────────────────────────────────────────────────────
if st.session_state.agent_result:
    r = st.session_state.agent_result
    st.info(f"🤖 **Agent insight (Round {st.session_state.round_num - 1}):** {r['reasoning']}")
    st.progress(r["confidence"], text=f"Agent confidence: {r['confidence']:.0%}")
    st.divider()

# ── Recommendations ───────────────────────────────────────────────────────────
st.subheader("Top Recommendations")
st.caption("👍 Like or 👎 Dislike songs, then click **Update My Taste** to improve results.")

try:
    results = get_recommendations(
        st.session_state.current_prefs,
        retriever,
        k=k,
        round_num=st.session_state.round_num,
        min_popularity=popularity_min,
    )
except Exception as exc:
    st.error(f"Error fetching recommendations: {exc}")
    results = []

if not results:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;">
      <svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"
           style="margin-bottom:16px;opacity:0.4;">
        <circle cx="32" cy="32" r="31" stroke="#7c3aed" stroke-width="1.5"/>
        <circle cx="32" cy="32" r="20" stroke="#4c1d95" stroke-width="1"/>
        <circle cx="32" cy="32" r="4"  fill="#7c3aed"/>
        <line x1="32" y1="12" x2="32" y2="8"  stroke="#7c3aed" stroke-width="2" stroke-linecap="round"/>
        <line x1="32" y1="56" x2="32" y2="52" stroke="#7c3aed" stroke-width="2" stroke-linecap="round"/>
        <line x1="12" y1="32" x2="8"  y2="32" stroke="#7c3aed" stroke-width="2" stroke-linecap="round"/>
        <line x1="56" y1="32" x2="52" y2="32" stroke="#7c3aed" stroke-width="2" stroke-linecap="round"/>
      </svg>
      <div style="font-size:18px;font-weight:600;color:#a78bfa;margin-bottom:8px;">
        Nothing matched your vibe
      </div>
      <div style="font-size:14px;color:#6b7280;max-width:320px;margin:0 auto;">
        Try lowering <strong style="color:#c4b5fd;">Min Popularity</strong> in the sidebar,
        or loosen your genre and mood preferences.
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for rank, (song, score, explanation) in enumerate(results, start=1):
        song_id        = song["id"]
        current_rating = st.session_state.feedback.get(song_id, {}).get("rating")

        # Render the visual card (HTML) — Streamlit buttons can't live inside
        # raw HTML, so Like/Dislike stay as native widgets directly below.
        st.markdown(
            _song_card(rank, song, score, explanation, current_rating),
            unsafe_allow_html=True,
        )

        col_like, col_dislike, _ = st.columns([1, 1, 5])
        liked_label    = "✅ Liked"    if current_rating == "like"    else "👍 Like"
        disliked_label = "❌ Disliked" if current_rating == "dislike" else "👎 Dislike"

        with col_like:
            if st.button(liked_label, key=f"like_{song_id}_{st.session_state.round_num}"):
                st.session_state.feedback[song_id] = {"song": song, "rating": "like"}
                st.rerun()
        with col_dislike:
            if st.button(disliked_label, key=f"dislike_{song_id}_{st.session_state.round_num}"):
                st.session_state.feedback[song_id] = {"song": song, "rating": "dislike"}
                st.rerun()

        st.write("")  # breathing room between cards

# ── Feedback summary + update button ─────────────────────────────────────────
if st.session_state.feedback:
    st.subheader("Your Ratings This Round")

    liked_titles    = [v["song"]["title"] for v in st.session_state.feedback.values() if v["rating"] == "like"]
    disliked_titles = [v["song"]["title"] for v in st.session_state.feedback.values() if v["rating"] == "dislike"]

    if liked_titles:
        st.success(f"👍 Liked: {', '.join(liked_titles)}")
    if disliked_titles:
        st.error(f"👎 Disliked: {', '.join(disliked_titles)}")

    if st.button("🔁 Update My Taste", type="primary", use_container_width=True):
        with st.spinner("AI agent is analysing your feedback..."):
            try:
                feedback_items = list(st.session_state.feedback.values())
                liked_count    = sum(1 for f in feedback_items if f["rating"] == "like")

                st.session_state.evaluator.record_round(
                    round_num  = st.session_state.round_num,
                    shown      = len(results),
                    liked      = liked_count,
                    confidence = 0.0,
                )

                result = run_feedback_agent(
                    current_prefs  = st.session_state.current_prefs,
                    feedback_items = feedback_items,
                    round_num      = st.session_state.round_num,
                )

                st.session_state.evaluator.history[-1]["confidence"] = result["confidence"]

                # Carry forward all previously liked IDs plus this round's likes
                # so those songs are excluded from every future round.
                prev_liked   = st.session_state.current_prefs.get("liked_song_ids", [])
                newly_liked  = [v["song"]["id"] for v in feedback_items if v["rating"] == "like"]
                result["updated_prefs"]["liked_song_ids"] = list(set(prev_liked + newly_liked))

                prev_disliked_genres  = st.session_state.current_prefs.get("disliked_genres", [])
                newly_disliked_genres = [v["song"]["genre"] for v in feedback_items if v["rating"] == "dislike"]
                result["updated_prefs"]["disliked_genres"] = list(set(prev_disliked_genres + newly_disliked_genres))

                st.session_state.current_prefs = result["updated_prefs"]
                st.session_state.agent_result  = result
                st.session_state.round_num    += 1
                st.session_state.feedback      = {}
                st.rerun()

            except Exception as exc:
                st.error(f"Agent error: {exc}")

# ── Satisfaction trend chart ──────────────────────────────────────────────────
if len(st.session_state.evaluator.history) >= 1:
    st.divider()
    st.subheader("📈 How Your Recommendations Are Improving")
    chart_data = st.session_state.evaluator.as_chart_data()

    import pandas as pd
    df = pd.DataFrame({
        "Satisfaction %": chart_data["Satisfaction %"],
        "Confidence %":   chart_data["Confidence %"],
    }, index=chart_data["Round"])
    st.line_chart(df)
    st.caption(st.session_state.evaluator.summary())
