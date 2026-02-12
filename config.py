"""
WeEvolve configuration -- all paths resolved here, zero hardcoding.
=================================================================
Every module imports paths from here. No hardcoded /Users/... anywhere.

Data directory priority:
  1. WEEVOLVE_DATA_DIR env var
  2. ~/.weevolve/

Base directory (for integration modules scanning codebase):
  1. WEEVOLVE_BASE_DIR env var
  2. Current working directory
"""

import os
import json
from pathlib import Path


def get_data_dir() -> Path:
    """Resolve the WeEvolve data directory."""
    env_dir = os.environ.get('WEEVOLVE_DATA_DIR')
    if env_dir:
        return Path(env_dir)
    return Path.home() / '.weevolve'


def get_base_dir() -> Path:
    """Resolve the project base directory (for integration modules)."""
    env_dir = os.environ.get('WEEVOLVE_BASE_DIR')
    if env_dir:
        return Path(env_dir)
    return Path.cwd()


# Core data paths
DATA_DIR = get_data_dir()
BASE_DIR = get_base_dir()

WEEVOLVE_DB = DATA_DIR / 'weevolve.db'
EVOLUTION_STATE_PATH = DATA_DIR / 'weevolve_state.json'
EVOLUTION_LOG_PATH = DATA_DIR / 'evolution_log.jsonl'
BOOKMARKS_DIR = DATA_DIR / 'bookmarks'

# Genesis paths
GENESIS_DB_DEFAULT = DATA_DIR / 'genesis.db'
GENESIS_CURATED_DB_DEFAULT = DATA_DIR / 'genesis-curated.db'

# Integration module paths
QUALIFY_CACHE = DATA_DIR / 'qualify_cache.json'
EXPLORE_CACHE = DATA_DIR / 'explore_cache.json'
EXPLORE_LOG = DATA_DIR / 'explore_log.jsonl'
PLANS_DIR = DATA_DIR / 'integration_plans'
PLANS_INDEX = PLANS_DIR / 'index.json'
COST_LOG = DATA_DIR / 'integrate_cost.json'
TOOLS_DIR = BASE_DIR / 'tools'
INVENTORY_CACHE = DATA_DIR / 'inventory_cache.json'

# Ensure data directory exists on import
DATA_DIR.mkdir(parents=True, exist_ok=True)


def bootstrap_genesis():
    """Copy bundled genesis DB to user's data dir on first install."""
    user_genesis = GENESIS_CURATED_DB_DEFAULT
    if user_genesis.exists():
        return  # Already bootstrapped

    bundled = Path(__file__).parent / "data" / "genesis-curated.db"
    if bundled.exists():
        import shutil
        shutil.copy2(bundled, user_genesis)
        print(f"  Bootstrapped {user_genesis.stat().st_size // 1024}KB genesis knowledge")


def load_api_key():
    """
    Load Anthropic API key.
    Priority:
      1. ANTHROPIC_API_KEY env var (already set)
      2. ~/.weevolve/credentials.json
    """
    if os.environ.get('ANTHROPIC_API_KEY'):
        return

    creds_path = DATA_DIR / 'credentials.json'
    if creds_path.exists():
        try:
            with open(creds_path) as f:
                creds = json.load(f)
            api_key = creds.get('anthropic', {}).get('api_key', '')
            if api_key:
                os.environ['ANTHROPIC_API_KEY'] = api_key
        except Exception:
            pass


# Auto-bootstrap genesis on first import
bootstrap_genesis()
