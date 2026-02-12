"""
WeEvolve license gating -- local file check for pro tier features.
==================================================================
v1: No server validation. License stored at ~/.weevolve/license.json.
Future: webhook validation on activate + periodic refresh.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from . import config

LICENSE_PATH = config.get_data_dir() / "license.json"

FREE_FEATURES = frozenset({
    "learn", "scan", "recall", "status",
    "quest", "genesis", "daemon", "voice",
})

PRO_FEATURES = frozenset({
    "chat", "companion", "8owls",
    "nats", "daemon_advanced", "glasses",
})

ALL_FEATURES = FREE_FEATURES | PRO_FEATURES


def _read_license() -> dict | None:
    """Read and return license data, or None if missing/corrupt."""
    if not LICENSE_PATH.exists():
        return None
    try:
        with open(LICENSE_PATH) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        if not data.get("key", "").startswith("we-pro-"):
            return None
        if data.get("tier") != "pro":
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def is_pro() -> bool:
    """Return True if a valid pro license exists on disk."""
    return _read_license() is not None


def get_tier() -> str:
    """Return 'pro' or 'free' based on local license file."""
    return "pro" if is_pro() else "free"


def check_feature(feature_name: str) -> bool:
    """Return True if the feature is available for the current tier."""
    if feature_name in FREE_FEATURES:
        return True
    if feature_name in PRO_FEATURES:
        return is_pro()
    return False


def show_upgrade_prompt(feature: str) -> None:
    """Print a brief, non-annoying upgrade nudge."""
    print(
        f"\n  '{feature}' requires WeEvolve Pro ($7.99/mo)\n"
        "  Unlock: 8 Owls, 3D Companion, Background Agents, Voice Chat\n"
        "  -> https://weevolve.ai/pro\n"
    )


def activate_license(key: str, email: str = "") -> bool:
    """Save a license key to disk. Returns True on success."""
    if not key.startswith("we-pro-"):
        print("  Invalid license key format. Expected: we-pro-xxxx")
        return False

    license_data = {
        "key": key,
        "email": email,
        "tier": "pro",
        "activated_at": datetime.now(timezone.utc).isoformat(),
    }

    LICENSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(LICENSE_PATH, "w") as f:
            json.dump(license_data, f, indent=2)
        print(f"  Pro license activated for {email or 'unknown'}")
        return True
    except OSError as err:
        print(f"  Failed to save license: {err}")
        return False
