"""
Evaluator — tracks recommendation quality across feedback rounds.

Records satisfaction per round (liked / total shown) and detects whether
the system is actually improving. This data is displayed in the app as a
trend chart and included in the test harness report.

Usage
-----
    from src.evaluator import Evaluator

    ev = Evaluator()
    ev.record_round(round_num=1, shown=5, liked=2, confidence=0.7)
    ev.record_round(round_num=2, shown=5, liked=4, confidence=0.85)

    print(ev.summary())
    print(ev.is_improving())   # True
"""

from src.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Tracks per-round satisfaction scores and improvement trend."""

    def __init__(self):
        self.history: list[dict] = []   # list of round records

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_round(
        self,
        round_num: int,
        shown: int,
        liked: int,
        confidence: float = 0.0,
    ) -> None:
        """
        Record the outcome of one recommendation round.

        Parameters
        ----------
        round_num  : which round (1, 2, 3 …)
        shown      : how many songs were shown to the user
        liked      : how many the user rated as liked
        confidence : agent confidence score for this round (0–1)
        """
        satisfaction = round(liked / shown, 2) if shown > 0 else 0.0

        record = {
            "round":        round_num,
            "shown":        shown,
            "liked":        liked,
            "satisfaction": satisfaction,
            "confidence":   confidence,
        }
        self.history.append(record)

        logger.info(
            "Round %d recorded | liked=%d/%d | satisfaction=%.0f%% | confidence=%.2f",
            round_num, liked, shown, satisfaction * 100, confidence,
        )

    # ── Analysis ──────────────────────────────────────────────────────────────

    def is_improving(self) -> bool:
        """
        Returns True if the latest round's satisfaction is higher than
        the first round's satisfaction (requires at least 2 rounds).
        """
        if len(self.history) < 2:
            return False
        return self.history[-1]["satisfaction"] > self.history[0]["satisfaction"]

    def satisfaction_scores(self) -> list[float]:
        """Returns list of satisfaction scores in round order."""
        return [r["satisfaction"] for r in self.history]

    def average_confidence(self) -> float:
        """Returns mean confidence score across all rounds."""
        if not self.history:
            return 0.0
        return round(sum(r["confidence"] for r in self.history) / len(self.history), 2)

    def best_round(self) -> dict:
        """Returns the round record with the highest satisfaction."""
        if not self.history:
            return {}
        return max(self.history, key=lambda r: r["satisfaction"])

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """
        Returns a human-readable summary string suitable for the test harness
        and the assignment write-up.

        Example output:
            3 rounds | Satisfaction: 40% → 60% → 80% | Avg confidence: 0.78 | Improving: Yes
        """
        if not self.history:
            return "No rounds recorded yet."

        scores = " → ".join(f"{int(s * 100)}%" for s in self.satisfaction_scores())
        improving = "Yes" if self.is_improving() else "No"

        return (
            f"{len(self.history)} round(s) | "
            f"Satisfaction: {scores} | "
            f"Avg confidence: {self.average_confidence()} | "
            f"Improving: {improving}"
        )

    def as_chart_data(self) -> dict:
        """
        Returns data in a format ready for st.line_chart().

        Example: {"Round": [1, 2, 3], "Satisfaction %": [40, 60, 80]}
        """
        return {
            "Round":           [r["round"]                    for r in self.history],
            "Satisfaction %":  [round(r["satisfaction"] * 100) for r in self.history],
            "Confidence %":    [round(r["confidence"]    * 100) for r in self.history],
        }
