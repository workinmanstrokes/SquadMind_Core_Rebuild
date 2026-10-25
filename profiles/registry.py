"""Game profile registry: auto-detects which game is on screen via a small
template-matched signature per profile, then delegates HUD analysis to it.

Adding a new game later: drop a signature crop in signatures/, add a module
with matches_signature(frame) and analyze(frame), and register it below.
"""
from . import nhl

PROFILES = {
    "nhl": nhl,
}


def detect_game(frame):
    """Returns the best-matching profile name, or None if nothing matches
    confidently enough."""
    best_name = None
    best_score = 0.0
    for name, module in PROFILES.items():
        score = module.matches_signature(frame)
        if score > best_score:
            best_score = score
            best_name = name
    if best_score >= getattr(PROFILES.get(best_name), "SIGNATURE_THRESHOLD", 0.7):
        return best_name
    return None


def analyze(frame, game_name):
    return PROFILES[game_name].analyze(frame)
