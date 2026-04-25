"""
Structured logger for the recommendation system.

Every agent call, feedback round, retrieval event, and error is written to
logs/session.log as a timestamped JSON line so the session is fully auditable.

Usage
-----
    from src.logger import get_logger
    log = get_logger(__name__)
    log.info("agent_call", round=1, confidence=0.82)
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any


# ── Paths ─────────────────────────────────────────────────────────────────────

LOG_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "session.log")


# ── JSON formatter ────────────────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    """Formats every log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts":      datetime.now(timezone.utc).isoformat(),
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
        }
        # Attach any extra kwargs passed to the log call
        for key, value in record.__dict__.items():
            if key not in (
                "args", "created", "exc_info", "exc_text", "filename",
                "funcName", "levelname", "levelno", "lineno", "message",
                "module", "msecs", "msg", "name", "pathname", "process",
                "processName", "relativeCreated", "stack_info", "taskName",
                "thread", "threadName",
            ):
                try:
                    json.dumps(value)
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = str(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


# ── Setup (called once at import time) ───────────────────────────────────────

def _setup() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)

    root = logging.getLogger()
    if root.handlers:
        return   # already configured

    root.setLevel(logging.DEBUG)

    # File handler — full JSON lines, everything
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_JsonFormatter())
    root.addHandler(fh)

    # Console handler — human-readable, INFO and above only
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(ch)


_setup()


# ── Public helper ─────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Returns a named logger pre-configured by this module."""
    return logging.getLogger(name)


# ── Structured event helpers ──────────────────────────────────────────────────
# These are called from other modules so every key event is logged the same way.

_ev = logging.getLogger("events")


def log_retrieval(round_num: int, query_prefs: dict, num_candidates: int) -> None:
    """Logged whenever the retriever runs a vector search."""
    _ev.info(
        "retrieval",
        extra={
            "round":          round_num,
            "num_candidates": num_candidates,
            "query_genre":    query_prefs.get("genre"),
            "query_mood":     query_prefs.get("mood"),
        },
    )


def log_agent_call(round_num: int, feedback_summary: str,
                   confidence: float, updated_prefs: dict) -> None:
    """Logged whenever the Claude agent processes feedback and updates prefs."""
    _ev.info(
        "agent_call",
        extra={
            "round":            round_num,
            "confidence":       confidence,
            "feedback_summary": feedback_summary,
            "updated_genre":    updated_prefs.get("genre"),
            "updated_mood":     updated_prefs.get("mood"),
        },
    )


def log_recommendations(round_num: int, titles: list, liked_count: int) -> None:
    """Logged after each recommendation round with how many the user liked."""
    _ev.info(
        "recommendations",
        extra={
            "round":       round_num,
            "titles":      titles,
            "liked_count": liked_count,
            "total":       len(titles),
        },
    )


def log_error(context: str, error: Exception) -> None:
    """Logged whenever an exception is caught anywhere in the system."""
    _ev.error(
        "error",
        extra={
            "context":    context,
            "error_type": type(error).__name__,
            "error_msg":  str(error),
        },
        exc_info=True,
    )
