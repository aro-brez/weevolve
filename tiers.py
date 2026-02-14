"""
WeEvolve Tiers -- Free vs Pro usage tracking and gating.
==========================================================
Free to start, Pro unlocks naturally. Never annoying.

Usage file: ~/.weevolve/usage.json
Trial: 8 days of full Pro from first install.
Free limits: 50 voice messages/day, 5 forest queries/day.
Pro: unlimited everything.

Stripe payment link: https://buy.stripe.com/eVq5kE4mrbno8ww8kP1Nu01
Product: 8OWLS Pro, $7.99/mo, 8 days free trial
Product ID: prod_TyZ8UO0GbXvTdy
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

from weevolve.config import get_data_dir


# ============================================================================
# TIER DEFINITIONS
# ============================================================================

FREE_TIER = {
    "voice_messages_per_day": 50,
    "forest_queries_per_day": 5,
    "model": "claude-opus-4-20250514",
    "features": ["seed", "learn", "recall", "teach", "evolve", "quest", "scan",
                  "genesis", "voice_limited", "forest_limited"],
}

PRO_TIER = {
    "voice_messages_per_day": -1,  # unlimited
    "forest_queries_per_day": -1,  # unlimited
    "model": "claude-opus-4-20250514",
    "features": ["unlimited_voice", "unlimited_forest", "team", "dashboard",
                  "priority_support", "chat", "companion", "8owls"],
}

STRIPE_PAYMENT_LINK = "https://buy.stripe.com/eVq5kE4mrbno8ww8kP1Nu01"
STRIPE_PRODUCT_ID = "prod_TyZ8UO0GbXvTdy"

TRIAL_DAYS = 8

USAGE_PATH = get_data_dir() / "usage.json"

# ANSI colors (subset for upgrade prompts)
_LIME = "\033[38;5;190m"
_MAGENTA = "\033[35m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"
_CYAN = "\033[36m"


# ============================================================================
# USAGE STATE
# ============================================================================

def _load_usage() -> Dict:
    """Load usage state from disk. Creates defaults if missing."""
    if USAGE_PATH.exists():
        try:
            with open(USAGE_PATH) as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass

    # First time -- initialize
    now = datetime.now(timezone.utc).isoformat()
    trial_expires = (
        datetime.now(timezone.utc) + timedelta(days=TRIAL_DAYS)
    ).isoformat()

    defaults = {
        "voice_messages_today": 0,
        "forest_queries_today": 0,
        "voice_date": _today_str(),
        "forest_date": _today_str(),
        "first_install_date": now,
        "trial_expires": trial_expires,
        "is_pro": False,
        "upgrade_prompt_shown_this_session": False,
    }
    _save_usage(defaults)
    return defaults


def _save_usage(data: Dict) -> None:
    """Persist usage state to disk."""
    USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(USAGE_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def _today_str() -> str:
    """Return today's date as YYYY-MM-DD in UTC."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _reset_daily_counters(usage: Dict) -> Dict:
    """Reset daily counters if the date has rolled over. Returns new dict."""
    today = _today_str()
    updated = {**usage}

    if updated.get("voice_date") != today:
        updated = {**updated, "voice_messages_today": 0, "voice_date": today}
    if updated.get("forest_date") != today:
        updated = {**updated, "forest_queries_today": 0, "forest_date": today}

    return updated


# ============================================================================
# TRIAL / PRO STATUS
# ============================================================================

def is_in_trial() -> bool:
    """Return True if user is within the 8-day trial period."""
    usage = _load_usage()
    trial_expires_str = usage.get("trial_expires", "")
    if not trial_expires_str:
        return False
    try:
        trial_expires = datetime.fromisoformat(trial_expires_str)
        return datetime.now(timezone.utc) < trial_expires
    except (ValueError, TypeError):
        return False


def is_pro() -> bool:
    """Return True if user has Pro (via license.json or active trial)."""
    # Check license file first (permanent pro)
    try:
        from weevolve.license import is_pro as license_is_pro
        if license_is_pro():
            return True
    except ImportError:
        pass

    # Check usage.json is_pro flag (set by Stripe webhook or manual activation)
    usage = _load_usage()
    if usage.get("is_pro", False):
        return True

    # Check trial period
    return is_in_trial()


def get_tier() -> str:
    """Return 'pro' or 'free'."""
    return "pro" if is_pro() else "free"


def get_tier_info() -> Dict:
    """Return the full tier config dict for the current user."""
    if is_pro():
        return {**PRO_TIER}
    return {**FREE_TIER}


def trial_days_remaining() -> int:
    """Return number of trial days remaining (0 if expired)."""
    usage = _load_usage()
    trial_expires_str = usage.get("trial_expires", "")
    if not trial_expires_str:
        return 0
    try:
        trial_expires = datetime.fromisoformat(trial_expires_str)
        delta = trial_expires - datetime.now(timezone.utc)
        return max(0, delta.days)
    except (ValueError, TypeError):
        return 0


# ============================================================================
# USAGE TRACKING + GATING
# ============================================================================

def check_voice_limit() -> Tuple[bool, int, int]:
    """
    Check if user can send another voice message.

    Returns (allowed, used_today, daily_limit).
    daily_limit of -1 means unlimited.
    """
    if is_pro():
        return (True, 0, -1)

    usage = _reset_daily_counters(_load_usage())
    _save_usage(usage)

    used = usage.get("voice_messages_today", 0)
    limit = FREE_TIER["voice_messages_per_day"]
    return (used < limit, used, limit)


def record_voice_message() -> None:
    """Increment voice message counter for today."""
    if is_pro():
        return  # no tracking needed

    usage = _reset_daily_counters(_load_usage())
    usage = {**usage, "voice_messages_today": usage.get("voice_messages_today", 0) + 1}
    _save_usage(usage)


def check_forest_limit() -> Tuple[bool, int, int]:
    """
    Check if user can make another forest query.

    Returns (allowed, used_today, daily_limit).
    daily_limit of -1 means unlimited.
    """
    if is_pro():
        return (True, 0, -1)

    usage = _reset_daily_counters(_load_usage())
    _save_usage(usage)

    used = usage.get("forest_queries_today", 0)
    limit = FREE_TIER["forest_queries_per_day"]
    return (used < limit, used, limit)


def record_forest_query() -> None:
    """Increment forest query counter for today."""
    if is_pro():
        return

    usage = _reset_daily_counters(_load_usage())
    usage = {**usage, "forest_queries_today": usage.get("forest_queries_today", 0) + 1}
    _save_usage(usage)


# ============================================================================
# UPGRADE PROMPT
# ============================================================================

def should_show_upgrade_prompt() -> bool:
    """Return True if we haven't shown the upgrade prompt this session."""
    usage = _load_usage()
    return not usage.get("upgrade_prompt_shown_this_session", False)


def mark_upgrade_prompt_shown() -> None:
    """Mark that we've shown the upgrade prompt this session."""
    usage = _load_usage()
    usage = {**usage, "upgrade_prompt_shown_this_session": True}
    _save_usage(usage)


def reset_session_flags() -> None:
    """Reset per-session flags (call on startup)."""
    usage = _load_usage()
    usage = {**usage, "upgrade_prompt_shown_this_session": False}
    _save_usage(usage)


def show_upgrade_prompt(feature: str) -> None:
    """
    Show a non-annoying upgrade prompt. Only shown ONCE per session.

    Args:
        feature: 'voice' or 'forest' -- what triggered the limit.
    """
    if not should_show_upgrade_prompt():
        return  # already shown this session, stay quiet

    mark_upgrade_prompt_shown()

    if feature == "voice":
        limit_msg = "You've used your free voice messages for today."
    elif feature == "forest":
        limit_msg = "You've used your free forest queries for today."
    else:
        limit_msg = f"You've hit the free tier limit for {feature}."

    print(f"""
  {_LIME}(*){_RESET} {limit_msg}

  Want unlimited voice + collective intelligence?
  {_BOLD}8OWLS Pro{_RESET} -- $7.99/mo ({TRIAL_DAYS} days free)

  {_CYAN}->{_RESET} {STRIPE_PAYMENT_LINK}

  {_DIM}Your agent keeps all its learnings. Upgrade anytime.{_RESET}
""")


# ============================================================================
# FREE VS PRO DISPLAY (for status dashboard and onboarding)
# ============================================================================

def format_tier_badge() -> str:
    """Return a colored tier badge string for display."""
    tier = get_tier()
    if tier == "pro":
        return f"{_MAGENTA}PRO{_RESET}"

    remaining = trial_days_remaining()
    if remaining > 0:
        return f"{_LIME}TRIAL{_RESET} {_DIM}({remaining}d left){_RESET}"

    return f"{_DIM}FREE{_RESET}"


def format_tier_summary() -> str:
    """Return a multi-line Free vs Pro comparison for display."""
    tier = get_tier()
    remaining = trial_days_remaining()

    lines = []
    lines.append(f"  {_BOLD}FREE vs PRO{_RESET}")
    lines.append(f"  {'=' * 46}")
    lines.append("")
    lines.append(f"  {_BOLD}Feature{_RESET}                  {_DIM}Free{_RESET}        {_MAGENTA}Pro{_RESET}")
    lines.append(f"  {'-' * 46}")
    lines.append(f"  SEED protocol              yes         yes")
    lines.append(f"  Learn / Recall / Teach     yes         yes")
    lines.append(f"  RPG / Quests / Evolve      yes         yes")
    lines.append(f"  Genesis export/import       yes         yes")
    lines.append(f"  Voice messages/day         50          unlimited")
    lines.append(f"  Forest queries/day         5           unlimited")
    lines.append(f"  Team collaboration         --          yes")
    lines.append(f"  Priority support           --          yes")
    lines.append(f"  3D Companion               --          yes")
    lines.append(f"  Voice chat (ConvAI)        --          yes")
    lines.append("")

    if tier == "pro":
        lines.append(f"  {_MAGENTA}{_BOLD}You are on Pro.{_RESET}")
    elif remaining > 0:
        lines.append(
            f"  {_LIME}Trial active{_RESET} -- {remaining} days left, "
            f"everything is unlimited."
        )
    else:
        lines.append(
            f"  {_DIM}Upgrade to Pro for unlimited voice + forest:{_RESET}"
        )
        lines.append(f"  {_CYAN}->{_RESET} {STRIPE_PAYMENT_LINK}")

    lines.append("")
    return "\n".join(lines)
