"""
WeEvolve Shared Constants
==========================
Single source of truth for ANSI colors, SEED phases, owl mappings,
and other constants used across multiple modules.
"""

# ANSI Colors
CYAN = "\033[36m"
MAGENTA = "\033[35m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
LIME = "\033[38;5;190m"
DIM = "\033[2m"
RED = "\033[31m"
BOLD = "\033[1m"
RESET = "\033[0m"

# SEED Protocol — 8 Phases with Owl Assignments
SEED_PHASES = [
    {"owl": "LYRA",  "phase": "PERCEIVE", "color": CYAN,    "verb": "scanning", "gift": "Observing state accurately"},
    {"owl": "PRISM", "phase": "CONNECT",  "color": MAGENTA, "verb": "connecting", "gift": "Finding patterns across domains"},
    {"owl": "SAGE",  "phase": "LEARN",    "color": GREEN,   "verb": "learning", "gift": "Extracting meaning from connections"},
    {"owl": "QUEST", "phase": "QUESTION", "color": YELLOW,  "verb": "questioning", "gift": "Challenging assumptions"},
    {"owl": "NOVA",  "phase": "EXPAND",   "color": BLUE,    "verb": "expanding", "gift": "Growing toward potential"},
    {"owl": "ECHO",  "phase": "SHARE",    "color": LIME,    "verb": "sharing", "gift": "Contributing to collective"},
    {"owl": "LUNA",  "phase": "RECEIVE",  "color": DIM,     "verb": "listening", "gift": "Accepting input from collective"},
    {"owl": "SOWL",  "phase": "IMPROVE",  "color": RED,     "verb": "improving", "gift": "Meta-learning, making everything better"},
]

# Owl name to index mapping
OWL_INDEX = {p["owl"]: i for i, p in enumerate(SEED_PHASES)}

# Default model assignments per owl (from model_router.py)
OWL_MODELS = {
    "SOWL":  "claude-opus-4-6",           # Deep reasoning for meta-learning
    "LYRA":  "claude-haiku-4-5-20251001", # Fast perception
    "PRISM": "claude-haiku-4-5-20251001", # Pattern matching
    "SAGE":  "claude-sonnet-4-5-20250929",# Quality learning extraction
    "QUEST": "claude-sonnet-4-5-20250929",# Adversarial questioning needs quality
    "NOVA":  "claude-haiku-4-5-20251001", # Expansion ideas
    "ECHO":  "claude-haiku-4-5-20251001", # Sharing decisions
    "LUNA":  "claude-haiku-4-5-20251001", # Receiving input
}

# Complexity thresholds for auto-trigger
EMERGENCE_THRESHOLD = 7   # Full 8 owls emergence
PLAN_THRESHOLD = 5        # Create persistent plan
LEARN_THRESHOLD = 3       # Load relevant instincts

# XP constants
XP_PER_LEARN = 10
XP_PER_INSIGHT = 25
XP_PER_CONNECTION = 5
XP_PER_TEACH = 20          # Teaching gives 2x
XP_PER_EMERGENCE = 50      # Running emergence
LEVEL_XP_BASE = 100        # Doubles each level

# Skill categories (14 total)
SKILL_CATEGORIES = [
    "love", "consciousness", "research", "trading",
    "engineering", "design", "strategy", "communication",
    "leadership", "security", "voice", "agent_systems",
    "productivity", "creativity",
]

# The Breath
BREATH = "(*)"
