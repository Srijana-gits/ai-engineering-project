"""
Feedback Agent — multi-step agentic workflow using Google Gemini.

When the user rates songs (like / dislike) the agent runs three observable
tool-call steps in sequence:

  1. analyze_feedback   — identifies taste patterns in the ratings
  2. update_preferences — translates patterns into updated numeric prefs
  3. set_confidence     — rates how sure it is (0.0 – 1.0)

Each step is logged so the reasoning chain is fully visible.

Usage
-----
    from src.feedback_agent import run_feedback_agent

    result = run_feedback_agent(
        current_prefs = { "genre": "rock", "energy": 0.6, ... },
        feedback_items = [
            {"song": song_dict, "rating": "like"},
            {"song": song_dict, "rating": "dislike"},
        ],
        round_num = 1,
    )
    # result["updated_prefs"]  -> new preference dict
    # result["confidence"]     -> float 0-1
    # result["reasoning"]      -> human-readable insight string
"""

import json
import os

import google.generativeai as genai
from dotenv import load_dotenv

from src.logger import get_logger, log_agent_call, log_error

load_dotenv()

logger = get_logger(__name__)

# ── Gemini client ─────────────────────────────────────────────────────────────

genai.configure(api_key=os.getenv("GEMINI_API_KEY", "").strip())
MODEL = "gemini-2.0-flash"

# ── Valid values the model must choose from ───────────────────────────────────

VALID_GENRES = [
    "acoustic", "afrobeat", "alt-rock", "alternative", "ambient", "blues",
    "classical", "club", "country", "dance", "dancehall", "disco", "drum-and-bass",
    "dub", "dubstep", "edm", "electronic", "emo", "folk", "funk", "gospel",
    "goth", "grunge", "hard-rock", "hardcore", "heavy-metal", "hip-hop", "house",
    "indie", "indie-pop", "jazz", "k-pop", "latin", "metal", "opera", "piano",
    "pop", "punk", "r-n-b", "reggae", "reggaeton", "rock", "soul", "techno",
    "trance", "trip-hop",
]

VALID_MOODS = [
    "angry", "chill", "energetic", "focused", "happy",
    "intense", "melancholic", "moody", "relaxed", "sad",
]

# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS = [
    genai.protos.Tool(function_declarations=[
        genai.protos.FunctionDeclaration(
            name="analyze_feedback",
            description=(
                "STEP 1: Analyze the user's liked and disliked songs to identify "
                "clear taste patterns. Look at genre, mood, energy, valence, and "
                "other audio features."
            ),
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "liked_patterns":   genai.protos.Schema(type=genai.protos.Type.STRING, description="What do the liked songs have in common?"),
                    "disliked_patterns":genai.protos.Schema(type=genai.protos.Type.STRING, description="What do the disliked songs have in common?"),
                    "key_insight":      genai.protos.Schema(type=genai.protos.Type.STRING, description="One concise sentence summarising what this user actually wants."),
                },
                required=["liked_patterns", "disliked_patterns", "key_insight"],
            ),
        ),
        genai.protos.FunctionDeclaration(
            name="update_preferences",
            description=(
                "STEP 2: Translate the taste analysis into updated numeric preference "
                "values. All float fields must be between 0.0 and 1.0. "
                f"genre must be one of: {', '.join(VALID_GENRES)}. "
                f"mood must be one of: {', '.join(VALID_MOODS)}."
            ),
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "genre":            genai.protos.Schema(type=genai.protos.Type.STRING),
                    "mood":             genai.protos.Schema(type=genai.protos.Type.STRING),
                    "energy":           genai.protos.Schema(type=genai.protos.Type.NUMBER),
                    "valence":          genai.protos.Schema(type=genai.protos.Type.NUMBER),
                    "danceability":     genai.protos.Schema(type=genai.protos.Type.NUMBER),
                    "acousticness":     genai.protos.Schema(type=genai.protos.Type.NUMBER),
                    "instrumentalness": genai.protos.Schema(type=genai.protos.Type.NUMBER),
                    "speechiness":      genai.protos.Schema(type=genai.protos.Type.NUMBER),
                    "tempo_bpm":        genai.protos.Schema(type=genai.protos.Type.NUMBER),
                },
                required=[
                    "genre", "mood", "energy", "valence", "danceability",
                    "acousticness", "instrumentalness", "speechiness", "tempo_bpm",
                ],
            ),
        ),
        genai.protos.FunctionDeclaration(
            name="set_confidence",
            description=(
                "STEP 3: Rate your confidence in the preference update. "
                "Use a low score (0.3–0.5) if there are very few ratings or they conflict. "
                "Use a high score (0.8–1.0) if patterns are clear and consistent."
            ),
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "score":  genai.protos.Schema(type=genai.protos.Type.NUMBER, description="Confidence score between 0.0 and 1.0."),
                    "reason": genai.protos.Schema(type=genai.protos.Type.STRING, description="One sentence explaining this confidence level."),
                },
                required=["score", "reason"],
            ),
        ),
    ])
]

# ── Prompt builder ─────────────────────────────────────────────────────────────

def _build_prompt(current_prefs: dict, feedback_items: list) -> str:
    liked    = [f for f in feedback_items if f["rating"] == "like"]
    disliked = [f for f in feedback_items if f["rating"] == "dislike"]

    def fmt(items):
        lines = []
        for f in items:
            s = f["song"]
            lines.append(
                f"  - \"{s['title']}\" by {s['artist']} "
                f"(genre={s['genre']}, mood={s['mood']}, "
                f"energy={s['energy']}, valence={s['valence']}, "
                f"danceability={s['danceability']}, acousticness={s['acousticness']}, "
                f"tempo={s['tempo_bpm']} BPM)"
            )
        return "\n".join(lines) if lines else "  (none)"

    return f"""You are a music taste analyst helping personalise a song recommender.

Current user preferences:
  genre={current_prefs.get('genre')}, mood={current_prefs.get('mood')},
  energy={current_prefs.get('energy')}, valence={current_prefs.get('valence')},
  danceability={current_prefs.get('danceability')}, acousticness={current_prefs.get('acousticness')},
  instrumentalness={current_prefs.get('instrumentalness')}, speechiness={current_prefs.get('speechiness')},
  tempo_bpm={current_prefs.get('tempo_bpm')}

Liked songs:
{fmt(liked)}

Disliked songs:
{fmt(disliked)}

You must call the three tools IN ORDER:
  1. analyze_feedback
  2. update_preferences
  3. set_confidence

Do not skip any step."""


# ── Agentic loop ───────────────────────────────────────────────────────────────

def run_feedback_agent(
    current_prefs: dict,
    feedback_items: list,
    round_num: int = 1,
) -> dict:
    """
    Runs the 3-step agentic feedback loop and returns updated preferences.

    Returns
    -------
    {
        "updated_prefs": dict,
        "confidence":    float,
        "reasoning":     str,
    }
    """
    if not feedback_items:
        logger.warning("run_feedback_agent called with no feedback items — returning prefs unchanged")
        return {"updated_prefs": current_prefs, "confidence": 0.0, "reasoning": "No feedback provided."}

    model = genai.GenerativeModel(
        model_name=MODEL,
        tools=TOOLS,
        system_instruction="You are a music taste analyst. Always use the provided tools to reason step by step.",
    )

    chat    = model.start_chat()
    steps   = {}
    prompt  = _build_prompt(current_prefs, feedback_items)
    message = prompt

    MAX_ITERATIONS = 8
    logger.info("Starting feedback agent | round=%d | feedback_count=%d", round_num, len(feedback_items))

    for iteration in range(MAX_ITERATIONS):

        if all(k in steps for k in ("analyze_feedback", "update_preferences", "set_confidence")):
            break

        try:
            response = chat.send_message(
                message,
                tool_config={"function_calling_config": {"mode": "ANY"}},
            )
        except Exception as exc:
            log_error("gemini_api_call", exc)
            raise

        # Collect all function calls from the response
        function_calls = [p.function_call for p in response.parts if p.function_call]

        if not function_calls:
            logger.warning("No tool call returned on iteration %d — breaking", iteration)
            break

        # Process each tool call and build the function responses
        tool_responses = []
        for fc in function_calls:
            name = fc.name
            args = dict(fc.args)
            steps[name] = args
            logger.info("Tool called: %s | args=%s", name, json.dumps(args, ensure_ascii=False, default=str))

            tool_responses.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=name,
                        response={"result": args},
                    )
                )
            )

        # Feed all results back in one message
        message = genai.protos.Content(parts=tool_responses, role="user")

    # ── Extract results ───────────────────────────────────────────────────────

    analysis  = steps.get("analyze_feedback",   {})
    new_prefs = steps.get("update_preferences", {})
    conf_step = steps.get("set_confidence",     {})

    updated_prefs = {**current_prefs, **new_prefs}

    for key in ["energy", "valence", "danceability", "acousticness",
                "instrumentalness", "speechiness"]:
        if key in updated_prefs:
            updated_prefs[key] = round(max(0.0, min(1.0, float(updated_prefs[key]))), 3)

    if "tempo_bpm" in updated_prefs:
        updated_prefs["tempo_bpm"] = round(max(40.0, min(250.0, float(updated_prefs["tempo_bpm"]))), 1)

    if updated_prefs.get("genre") not in VALID_GENRES:
        updated_prefs["genre"] = current_prefs.get("genre", "pop")

    if updated_prefs.get("mood") not in VALID_MOODS:
        updated_prefs["mood"] = current_prefs.get("mood", "happy")

    confidence = float(conf_step.get("score", 0.5))
    confidence = round(max(0.0, min(1.0, confidence)), 2)

    reasoning = analysis.get("key_insight", "Preferences updated based on your feedback.")

    log_agent_call(
        round_num        = round_num,
        feedback_summary = f"{sum(1 for f in feedback_items if f['rating']=='like')} likes, "
                           f"{sum(1 for f in feedback_items if f['rating']=='dislike')} dislikes",
        confidence       = confidence,
        updated_prefs    = updated_prefs,
    )

    return {
        "updated_prefs": updated_prefs,
        "confidence":    confidence,
        "reasoning":     reasoning,
    }
