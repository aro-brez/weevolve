#!/usr/bin/env python3
"""
Genesis Update Pipeline for WeEvolve
=====================================
Continuously updates genesis-curated.db with the latest high-quality discoveries.

The genesis database is the collective knowledge that every new user gets on install.
This module keeps it fresh so users never start with stale January knowledge.

Pipeline:
  1. Read current genesis-curated.db (get existing atom IDs)
  2. Read new high-quality atoms from main WeEvolve DB (quality >= 0.8)
  3. Read TinyFish scan results (if available)
  4. Deduplicate against existing genesis atoms (by ID + content hash)
  5. PII-strip and insert new atoms
  6. Increment version number
  7. Record update metadata

Usage:
  python3 genesis_updater.py                    # Update genesis with new atoms
  python3 genesis_updater.py --stats            # Show update stats
  python3 genesis_updater.py --min-quality 0.7  # Custom quality threshold
  python3 genesis_updater.py --dry-run          # Preview without writing

Integration:
  weevolve genesis update                       # From CLI
  weevolve update                               # Auto-checks genesis freshness

(C) LIVE FREE = LIVE FOREVER
"""

import sqlite3
import json
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from weevolve.config import (
    WEEVOLVE_DB, GENESIS_CURATED_DB_DEFAULT, DATA_DIR, load_api_key,
)

# Constants
DEFAULT_MIN_QUALITY = 0.8
GENESIS_VERSION_KEY = "genesis_version"
TINYFISH_RESULTS_DIR = DATA_DIR / "tinyfish_results"

# ANSI colors
CYAN = "\033[36m"
MAGENTA = "\033[35m"
GREEN_C = "\033[32m"
YELLOW_C = "\033[33m"
LIME_C = "\033[38;5;190m"
DIM_C = "\033[2m"
RED_C = "\033[31m"
BOLD_C = "\033[1m"
RESET_C = "\033[0m"

# PII patterns (same as core.py to avoid circular imports)
import re
PII_PATTERNS = [
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '[EMAIL]'),
    (re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'), '[IP]'),
    (re.compile(r'(?:sk-|key-|api_)[A-Za-z0-9_-]{20,}'), '[API_KEY]'),
    (re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'), '[PHONE]'),
    (re.compile(r'/Users/\w+/'), '/Users/[USER]/'),
    (re.compile(r'C:\\Users\\\w+\\'), 'C:\\Users\\[USER]\\'),
]


def _strip_pii(text: str) -> str:
    """Remove personally identifiable information from text."""
    if not text:
        return text
    result = text
    for pattern, replacement in PII_PATTERNS:
        # Use lambda to avoid re.sub interpreting backslashes in replacement
        result = pattern.sub(lambda m: replacement, result)
    return result


def _content_hash(content: str) -> str:
    """Generate a hash for dedup."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _get_genesis_version(genesis_db: sqlite3.Connection) -> int:
    """Read the current genesis version number from metadata."""
    try:
        row = genesis_db.execute(
            "SELECT value FROM genesis_meta WHERE key = ?",
            (GENESIS_VERSION_KEY,)
        ).fetchone()
        if row:
            return int(row[0])
    except Exception:
        pass
    return 1


def _set_genesis_version(genesis_db: sqlite3.Connection, version: int):
    """Set the genesis version number in metadata."""
    genesis_db.execute(
        "INSERT OR REPLACE INTO genesis_meta (key, value) VALUES (?, ?)",
        (GENESIS_VERSION_KEY, str(version))
    )


def _get_existing_genesis_ids(genesis_db: sqlite3.Connection) -> set:
    """Get all atom IDs currently in the genesis database."""
    rows = genesis_db.execute("SELECT id FROM genesis_atoms").fetchall()
    return {row[0] for row in rows}


def _get_existing_genesis_hashes(genesis_db: sqlite3.Connection) -> set:
    """Get content hashes for fuzzy dedup (based on learn field)."""
    rows = genesis_db.execute("SELECT learn FROM genesis_atoms WHERE learn IS NOT NULL").fetchall()
    return {_content_hash(row[0]) for row in rows if row[0]}


def _load_tinyfish_results() -> List[Dict]:
    """Load any pending TinyFish scan results that should feed into genesis.

    TinyFish results are stored in DATA_DIR/tinyfish_results/ as JSON files.
    Each file contains extracted knowledge from web scans.
    """
    results = []
    if not TINYFISH_RESULTS_DIR.exists():
        return results

    for result_file in sorted(TINYFISH_RESULTS_DIR.glob("*.json")):
        try:
            with open(result_file) as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("quality", 0) >= DEFAULT_MIN_QUALITY:
                results.append(data)
        except (json.JSONDecodeError, OSError):
            continue

    return results


def update_genesis(
    min_quality: float = DEFAULT_MIN_QUALITY,
    include_tinyfish: bool = True,
    dry_run: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Update genesis-curated.db with new high-quality atoms from the main DB.

    Steps:
      1. Open genesis-curated.db and main weevolve.db
      2. Get existing genesis atom IDs for dedup
      3. Query new atoms from main DB (quality >= min_quality OR is_alpha)
      4. Optionally include TinyFish scan results
      5. Deduplicate by ID + content hash
      6. PII-strip and insert new atoms
      7. Increment version
      8. Update metadata

    Args:
        min_quality: Minimum quality threshold (default 0.8)
        include_tinyfish: Include TinyFish scan results
        dry_run: Preview only, don't write to database
        verbose: Print progress

    Returns:
        Dict with update stats
    """
    genesis_path = GENESIS_CURATED_DB_DEFAULT
    source_path = WEEVOLVE_DB

    if not source_path.exists():
        if verbose:
            print(f"  {RED_C}[ERROR]{RESET_C} Main WeEvolve DB not found: {source_path}")
        return {"error": "Source DB not found", "added": 0}

    # Ensure genesis exists (create from scratch if needed)
    genesis_path.parent.mkdir(parents=True, exist_ok=True)
    is_new_genesis = not genesis_path.exists()

    genesis_db = sqlite3.connect(str(genesis_path))
    genesis_db.execute("PRAGMA journal_mode=WAL")

    # Ensure tables exist (idempotent)
    genesis_db.executescript("""
        CREATE TABLE IF NOT EXISTS genesis_atoms (
            id TEXT PRIMARY KEY,
            title TEXT,
            perceive TEXT,
            connect TEXT,
            learn TEXT,
            question TEXT,
            expand TEXT,
            share TEXT,
            receive TEXT,
            improve TEXT,
            skills TEXT,
            quality REAL,
            is_alpha INTEGER DEFAULT 0,
            alpha_type TEXT,
            key_entities TEXT,
            connections TEXT,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS genesis_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_genesis_quality ON genesis_atoms(quality DESC);
        CREATE INDEX IF NOT EXISTS idx_genesis_alpha ON genesis_atoms(is_alpha);
    """)

    # Get existing genesis state
    current_version = _get_genesis_version(genesis_db)
    existing_ids = _get_existing_genesis_ids(genesis_db)
    existing_hashes = _get_existing_genesis_hashes(genesis_db)
    original_count = len(existing_ids)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {BOLD_C}(*) GENESIS UPDATE PIPELINE{RESET_C}")
        print(f"{'='*60}")
        print(f"  Genesis DB: {genesis_path}")
        print(f"  Current version: {current_version}")
        print(f"  Existing atoms: {original_count}")
        print(f"  Min quality: {min_quality}")
        if dry_run:
            print(f"  {YELLOW_C}DRY RUN — no changes will be written{RESET_C}")
        print()

    # Query new atoms from main DB
    source_db = sqlite3.connect(str(source_path))
    source_db.execute("PRAGMA journal_mode=WAL")

    new_atoms = source_db.execute("""
        SELECT id, title, perceive, connect, learn, question, expand,
               share, receive, improve, skills, quality, is_alpha,
               alpha_type, key_entities, connections, created_at
        FROM knowledge_atoms
        WHERE quality >= ? OR is_alpha = 1
        ORDER BY quality DESC
    """, (min_quality,)).fetchall()

    if verbose:
        print(f"  {CYAN}PERCEIVE{RESET_C} Found {len(new_atoms)} atoms in main DB (quality >= {min_quality})")

    # Collect TinyFish results
    tinyfish_atoms = []
    if include_tinyfish:
        tinyfish_results = _load_tinyfish_results()
        if tinyfish_results and verbose:
            print(f"  {MAGENTA}CONNECT{RESET_C} Found {len(tinyfish_results)} TinyFish scan results")
        tinyfish_atoms = tinyfish_results

    # Deduplicate and prepare inserts
    to_insert = []
    skipped_existing = 0
    skipped_hash = 0

    for atom in new_atoms:
        atom_id = atom[0]
        learn_text = atom[4] or ""

        # Skip if ID already in genesis
        if atom_id in existing_ids:
            skipped_existing += 1
            continue

        # Skip if content hash matches (fuzzy dedup on learn field)
        learn_hash = _content_hash(learn_text) if learn_text else ""
        if learn_hash and learn_hash in existing_hashes:
            skipped_hash += 1
            continue

        to_insert.append(atom)
        existing_ids.add(atom_id)
        if learn_hash:
            existing_hashes.add(learn_hash)

    # Process TinyFish atoms
    tinyfish_added = 0
    for tf_result in tinyfish_atoms:
        tf_id = tf_result.get("id") or _content_hash(
            json.dumps(tf_result, sort_keys=True)
        )
        if tf_id in existing_ids:
            continue

        learn_text = tf_result.get("learn", "")
        learn_hash = _content_hash(learn_text) if learn_text else ""
        if learn_hash and learn_hash in existing_hashes:
            continue

        # Convert TinyFish result to genesis atom tuple format
        tf_atom = (
            tf_id,
            tf_result.get("title", "TinyFish Discovery"),
            tf_result.get("perceive", ""),
            tf_result.get("connect", ""),
            tf_result.get("learn", ""),
            tf_result.get("question", ""),
            tf_result.get("expand", ""),
            tf_result.get("share", ""),
            tf_result.get("receive", ""),
            tf_result.get("improve", ""),
            json.dumps(tf_result.get("skills", [])),
            tf_result.get("quality", 0.8),
            1 if tf_result.get("is_alpha") else 0,
            tf_result.get("alpha_type"),
            json.dumps(tf_result.get("key_entities", [])),
            json.dumps(tf_result.get("connections", [])),
            tf_result.get("created_at", datetime.now(timezone.utc).isoformat()),
        )
        to_insert.append(tf_atom)
        existing_ids.add(tf_id)
        if learn_hash:
            existing_hashes.add(learn_hash)
        tinyfish_added += 1

    if verbose:
        print(f"  {GREEN_C}LEARN{RESET_C} Dedup complete: {len(to_insert)} new atoms to add")
        print(f"    Skipped (ID match): {skipped_existing}")
        print(f"    Skipped (content match): {skipped_hash}")
        if tinyfish_added > 0:
            print(f"    TinyFish discoveries: {tinyfish_added}")

    # Insert new atoms
    inserted = 0
    if not dry_run and to_insert:
        for atom in to_insert:
            try:
                genesis_db.execute("""
                    INSERT OR IGNORE INTO genesis_atoms (
                        id, title, perceive, connect, learn, question, expand,
                        share, receive, improve, skills, quality, is_alpha,
                        alpha_type, key_entities, connections, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    atom[0],
                    _strip_pii(atom[1]),   # title
                    _strip_pii(atom[2]),   # perceive
                    _strip_pii(atom[3]),   # connect
                    _strip_pii(atom[4]),   # learn
                    _strip_pii(atom[5]),   # question
                    _strip_pii(atom[6]),   # expand
                    _strip_pii(atom[7]),   # share
                    _strip_pii(atom[8]),   # receive
                    _strip_pii(atom[9]),   # improve
                    atom[10],              # skills (JSON, safe)
                    atom[11],              # quality
                    atom[12],              # is_alpha
                    atom[13],              # alpha_type
                    _strip_pii(atom[14]) if atom[14] else None,  # key_entities
                    _strip_pii(atom[15]) if atom[15] else None,  # connections
                    atom[16],              # created_at
                ))
                inserted += 1
            except Exception as e:
                if verbose:
                    print(f"  {DIM_C}[SKIP] Insert failed for {atom[0]}: {e}{RESET_C}")

        # Increment version
        new_version = current_version + 1
        _set_genesis_version(genesis_db, new_version)

        # Update metadata
        final_count = genesis_db.execute("SELECT COUNT(*) FROM genesis_atoms").fetchone()[0]
        avg_quality = genesis_db.execute("SELECT AVG(quality) FROM genesis_atoms").fetchone()[0] or 0
        alpha_count = genesis_db.execute(
            "SELECT COUNT(*) FROM genesis_atoms WHERE is_alpha = 1"
        ).fetchone()[0]

        meta_updates = {
            "genesis_version": str(new_version),
            "version": str(new_version),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "total_atoms": str(final_count),
            "avg_quality": f"{avg_quality:.3f}",
            "alpha_count": str(alpha_count),
            "pii_stripped": "True",
            "protocol": "SEED",
            "origin": "8OWLS",
            "tier": "curated",
            "min_quality": str(min_quality),
        }
        for key, value in meta_updates.items():
            genesis_db.execute(
                "INSERT OR REPLACE INTO genesis_meta (key, value) VALUES (?, ?)",
                (key, value)
            )

        genesis_db.commit()

        if verbose:
            print(f"\n  {LIME_C}EXPAND{RESET_C} Genesis updated successfully!")
            print(f"    Version: {current_version} -> {new_version}")
            print(f"    Atoms: {original_count} -> {final_count} (+{inserted})")
            print(f"    Avg quality: {avg_quality:.3f}")
            print(f"    Alpha count: {alpha_count}")
    elif dry_run and verbose:
        print(f"\n  {YELLOW_C}DRY RUN{RESET_C} Would insert {len(to_insert)} new atoms")
        print(f"    Version would become: {current_version + 1}")

    genesis_db.close()
    source_db.close()

    # Copy to bundled data directory for distribution
    if not dry_run and inserted > 0:
        bundled_path = Path(__file__).parent / "data" / "genesis-curated.db"
        if bundled_path.parent.exists():
            import shutil
            shutil.copy2(str(genesis_path), str(bundled_path))
            if verbose:
                print(f"  {GREEN_C}SHARE{RESET_C} Copied to distribution: {bundled_path}")

    stats = {
        "added": inserted,
        "skipped_existing": skipped_existing,
        "skipped_hash": skipped_hash,
        "tinyfish_added": tinyfish_added,
        "previous_count": original_count,
        "new_count": original_count + inserted,
        "previous_version": current_version,
        "new_version": current_version + 1 if inserted > 0 else current_version,
        "dry_run": dry_run,
    }

    if verbose:
        print(f"\n{'='*60}\n")

    return stats


def check_genesis_freshness() -> Dict:
    """
    Check if the user's local genesis is outdated compared to what's available.

    Returns dict with:
      - local_version: user's genesis version
      - latest_version: latest available version (from GitHub API or bundled)
      - needs_update: bool
      - atoms_behind: estimated new atoms available
    """
    genesis_path = GENESIS_CURATED_DB_DEFAULT

    # Get local version
    local_version = 0
    local_count = 0
    if genesis_path.exists():
        try:
            gdb = sqlite3.connect(str(genesis_path))
            local_version = _get_genesis_version(gdb)
            local_count = gdb.execute("SELECT COUNT(*) FROM genesis_atoms").fetchone()[0]
            gdb.close()
        except Exception:
            pass

    # Check bundled version (what came with this install)
    bundled_path = Path(__file__).parent / "data" / "genesis-curated.db"
    bundled_version = 0
    bundled_count = 0
    if bundled_path.exists():
        try:
            bdb = sqlite3.connect(str(bundled_path))
            bundled_version = _get_genesis_version(bdb)
            bundled_count = bdb.execute("SELECT COUNT(*) FROM genesis_atoms").fetchone()[0]
            bdb.close()
        except Exception:
            pass

    # Try GitHub API for latest release version
    latest_version = bundled_version
    latest_count = bundled_count
    try:
        import urllib.request
        url = "https://api.github.com/repos/aro-brez/weevolve/releases/latest"
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            release_data = json.loads(resp.read().decode())
            tag = release_data.get("tag_name", "")
            body = release_data.get("body", "")
            # Parse genesis version from release notes if present
            for line in body.split("\n"):
                if "genesis_version" in line.lower():
                    parts = line.split(":")
                    if len(parts) >= 2:
                        try:
                            latest_version = int(parts[1].strip())
                        except ValueError:
                            pass
    except Exception:
        pass  # Network unavailable, use bundled

    needs_update = bundled_version > local_version or bundled_count > local_count

    return {
        "local_version": local_version,
        "local_count": local_count,
        "bundled_version": bundled_version,
        "bundled_count": bundled_count,
        "latest_version": latest_version,
        "needs_update": needs_update,
        "atoms_behind": max(0, bundled_count - local_count),
    }


def auto_update_genesis(verbose: bool = True) -> Optional[Dict]:
    """
    Auto-update the user's genesis DB from bundled data if outdated.

    Called during `weevolve update` to give users the latest collective knowledge.

    Returns update stats if updated, None if already current.
    """
    freshness = check_genesis_freshness()

    if not freshness["needs_update"]:
        if verbose:
            print(f"  {GREEN_C}Genesis knowledge is current{RESET_C} "
                  f"(v{freshness['local_version']}, {freshness['local_count']} atoms)")
        return None

    if verbose:
        print(f"  {LIME_C}New collective knowledge available!{RESET_C}")
        print(f"    Local: v{freshness['local_version']} ({freshness['local_count']} atoms)")
        print(f"    Available: v{freshness['bundled_version']} ({freshness['bundled_count']} atoms)")
        print(f"    +{freshness['atoms_behind']} new knowledge atoms from the collective")

    # Import from bundled genesis
    bundled_path = Path(__file__).parent / "data" / "genesis-curated.db"
    if not bundled_path.exists():
        if verbose:
            print(f"  {DIM_C}Bundled genesis not found. Skipping auto-update.{RESET_C}")
        return None

    # Use the genesis_import from core to merge atoms
    try:
        from weevolve.core import genesis_import
        stats = genesis_import(str(bundled_path), verbose=verbose)
        return stats
    except Exception as e:
        if verbose:
            print(f"  {DIM_C}Auto-update failed: {e}{RESET_C}")
        return None


def main():
    """CLI entry point for genesis updater."""
    dry_run = "--dry-run" in sys.argv
    stats_only = "--stats" in sys.argv
    min_quality = DEFAULT_MIN_QUALITY

    if "--min-quality" in sys.argv:
        idx = sys.argv.index("--min-quality")
        if len(sys.argv) > idx + 1:
            try:
                min_quality = float(sys.argv[idx + 1])
            except ValueError:
                pass

    if stats_only:
        freshness = check_genesis_freshness()
        print(f"\n{'='*60}")
        print(f"  {BOLD_C}GENESIS FRESHNESS CHECK{RESET_C}")
        print(f"{'='*60}")
        print(f"  Local version: {freshness['local_version']}")
        print(f"  Local atoms: {freshness['local_count']}")
        print(f"  Bundled version: {freshness['bundled_version']}")
        print(f"  Bundled atoms: {freshness['bundled_count']}")
        print(f"  Needs update: {freshness['needs_update']}")
        if freshness['atoms_behind'] > 0:
            print(f"  {LIME_C}+{freshness['atoms_behind']} new atoms available{RESET_C}")
        print(f"{'='*60}\n")
        return

    update_genesis(
        min_quality=min_quality,
        include_tinyfish=True,
        dry_run=dry_run,
        verbose=True,
    )


if __name__ == "__main__":
    main()
