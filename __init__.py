"""
8 Owls Emergence System for WeEvolve
=====================================
Spawns 7 background agents with distinct owl personas for multi-perspective
analysis. SOWL (the caller) synthesizes all 7 into a final recommendation.

Usage:
    from weevolve.owls import emerge, quick_emerge

    result = emerge("Should we rewrite the auth module?")
    result = quick_emerge("Is this API design sound?")
"""

from weevolve.owls.emergence import emerge, quick_emerge
from weevolve.owls.synthesis import synthesize
from weevolve.owls.personas import OWL_PERSONAS, get_persona

__all__ = [
    "emerge",
    "quick_emerge",
    "synthesize",
    "OWL_PERSONAS",
    "get_persona",
]
