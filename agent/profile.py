"""Persistent user profile — stores distilled facts per session."""

import json
import os
from datetime import date

_PROFILES_DIR = os.path.join(os.path.dirname(__file__), "storage", "profiles")

_DEFAULT_PROFILE = {
    "name": None,
    "topics_explored": [],
    "preferences": None,
    "notes": None,
    "last_active": None,
}


def load_profile(session_id: str) -> dict:
    """Load the profile for *session_id*, returning defaults if none exists."""
    path = os.path.join(_PROFILES_DIR, f"{session_id}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {**_DEFAULT_PROFILE, "session_id": session_id}


def save_profile(session_id: str, profile: dict) -> None:
    """Persist *profile* to disk, stamping today's date."""
    os.makedirs(_PROFILES_DIR, exist_ok=True)
    profile["last_active"] = str(date.today())
    path = os.path.join(_PROFILES_DIR, f"{session_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
