#!/usr/bin/env python3
"""
WeEvolve INTEGRATE: qualify.py
================================
Score existing knowledge atoms for actionable GitHub repos.
Extract URLs. Zero API cost -- pure regex + heuristics.

Usage:
  python3 -m weevolve.qualify              # Qualify all atoms
  python3 -m weevolve.qualify --min-score 0.5  # Filter by min score
  python3 -m weevolve.qualify --limit 20   # Limit results
  python3 -m weevolve.qualify --json       # Output as JSON

(C) LIVE FREE = LIVE FOREVER
"""

import re
import json
import sqlite3
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

# Paths from shared config (no hardcoded paths)
from weevolve.config import WEEVOLVE_DB, QUALIFY_CACHE

# GitHub URL patterns
GITHUB_URL_PATTERN = re.compile(
    r'(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)(?:/[^\s)\]\"\']*)?',
    re.IGNORECASE
)

# Patterns that boost qualification score
SIGNAL_PATTERNS = {
    'tool_mention': re.compile(
        r'\b(?:tool|library|framework|sdk|cli|package|module|plugin|extension)\b',
        re.IGNORECASE
    ),
    'action_verb': re.compile(
        r'\b(?:build|deploy|integrate|install|clone|fork|use|implement|run|setup)\b',
        re.IGNORECASE
    ),
    'agent_related': re.compile(
        r'\b(?:agent|swarm|orchestrat|autonom|daemon|pipeline|workflow)\b',
        re.IGNORECASE
    ),
    'trading_related': re.compile(
        r'\b(?:trad|market|polymarket|signal|arbitrage|price|volume)\b',
        re.IGNORECASE
    ),
    'infra_related': re.compile(
        r'\b(?:nats|redis|postgres|sqlite|docker|kubernetes|api|websocket|mcp)\b',
        re.IGNORECASE
    ),
}

# Patterns that reduce score (noise)
NOISE_PATTERNS = {
    'promotional': re.compile(
        r'\b(?:giveaway|subscribe|follow me|retweet|like and share)\b',
        re.IGNORECASE
    ),
    'vaporware': re.compile(
        r'\b(?:coming soon|waitlist|pre-launch|stealth mode|announcing)\b',
        re.IGNORECASE
    ),
}


@dataclass
class QualifiedAtom:
    """A knowledge atom scored for integration potential."""
    atom_id: str
    title: str
    quality: float
    github_urls: List[str]
    qualification_score: float
    signals: Dict[str, int]
    repo_owner: str
    repo_name: str
    primary_url: str
    expand_text: str
    connect_text: str


def get_db() -> sqlite3.Connection:
    """Open read-only connection to WeEvolve DB."""
    if not WEEVOLVE_DB.exists():
        raise FileNotFoundError(f"WeEvolve DB not found: {WEEVOLVE_DB}")
    db = sqlite3.connect(str(WEEVOLVE_DB))
    db.row_factory = sqlite3.Row
    return db


def extract_github_urls(text: str) -> List[Tuple[str, str, str]]:
    """
    Extract GitHub repo URLs from text.
    Returns list of (full_url, owner, repo_name).
    """
    if not text:
        return []

    results = []
    seen = set()

    for match in GITHUB_URL_PATTERN.finditer(text):
        owner = match.group(1)
        repo = match.group(2)

        # Clean repo name (remove trailing punctuation, .git suffix)
        repo = re.sub(r'\.git$', '', repo)
        repo = re.sub(r'[.,;:!?]+$', '', repo)

        # Skip non-repo pages
        skip_owners = {'features', 'pricing', 'about', 'blog', 'explore', 'topics', 'trending'}
        if owner.lower() in skip_owners:
            continue

        canonical = f"github.com/{owner}/{repo}"
        if canonical.lower() not in seen:
            seen.add(canonical.lower())
            full_url = f"https://github.com/{owner}/{repo}"
            results.append((full_url, owner, repo))

    return results


def count_signals(text: str) -> Dict[str, int]:
    """Count signal pattern matches in text. Zero API cost."""
    counts = {}
    combined = text or ''

    for name, pattern in SIGNAL_PATTERNS.items():
        matches = pattern.findall(combined)
        if matches:
            counts[name] = len(matches)

    return counts


def count_noise(text: str) -> int:
    """Count noise pattern matches."""
    total = 0
    for pattern in NOISE_PATTERNS.values():
        total += len(pattern.findall(text or ''))
    return total


def compute_qualification_score(
    atom_quality: float,
    github_url_count: int,
    signals: Dict[str, int],
    noise_count: int,
    is_alpha: bool
) -> float:
    """
    Compute a 0-1 qualification score for integration potential.
    Combines atom quality, GitHub presence, signal strength, and noise penalty.
    """
    if github_url_count == 0:
        return 0.0

    # Base: atom quality contributes 40%
    base = atom_quality * 0.4

    # GitHub URL presence: 20% (1 URL = full, diminishing returns)
    url_score = min(1.0, github_url_count / 2.0) * 0.2

    # Signal strength: 30% (more relevant signals = higher score)
    total_signals = sum(signals.values())
    signal_score = min(1.0, total_signals / 8.0) * 0.3

    # Alpha bonus: 10%
    alpha_bonus = 0.1 if is_alpha else 0.0

    # Noise penalty
    noise_penalty = min(0.2, noise_count * 0.05)

    score = base + url_score + signal_score + alpha_bonus - noise_penalty
    return round(max(0.0, min(1.0, score)), 3)


def qualify_atoms(
    min_score: float = 0.3,
    limit: int = 50,
    use_cache: bool = True
) -> List[QualifiedAtom]:
    """
    Score all knowledge atoms for integration potential.
    Returns sorted list of QualifiedAtom (highest score first).
    Zero API cost -- pure local computation.
    """
    # Check cache
    if use_cache and QUALIFY_CACHE.exists():
        try:
            cache = json.loads(QUALIFY_CACHE.read_text())
            cache_age_seconds = __import__('time').time() - cache.get('timestamp', 0)
            # Cache valid for 1 hour
            if cache_age_seconds < 3600:
                cached_atoms = [QualifiedAtom(**a) for a in cache.get('atoms', [])]
                filtered = [a for a in cached_atoms if a.qualification_score >= min_score]
                return filtered[:limit]
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    db = get_db()

    rows = db.execute("""
        SELECT id, title, quality, is_alpha, raw_content,
               perceive, connect, learn, question, expand,
               share, receive, improve, source_url
        FROM knowledge_atoms
        ORDER BY quality DESC
    """).fetchall()

    qualified = []

    for row in rows:
        # Combine all text fields for analysis
        all_text = ' '.join(filter(None, [
            row['raw_content'] or '',
            row['perceive'] or '',
            row['connect'] or '',
            row['learn'] or '',
            row['question'] or '',
            row['expand'] or '',
            row['share'] or '',
            row['improve'] or '',
        ]))

        # Extract GitHub URLs
        github_urls = extract_github_urls(all_text)
        if not github_urls:
            continue

        # Count signals
        signals = count_signals(all_text)
        noise = count_noise(all_text)

        # Compute score
        score = compute_qualification_score(
            atom_quality=row['quality'] or 0.0,
            github_url_count=len(github_urls),
            signals=signals,
            noise_count=noise,
            is_alpha=bool(row['is_alpha'])
        )

        if score < min_score:
            continue

        # Pick the best (first) URL
        primary_url, owner, repo_name = github_urls[0]

        qualified.append(QualifiedAtom(
            atom_id=row['id'],
            title=row['title'] or 'Untitled',
            quality=row['quality'] or 0.0,
            github_urls=[u[0] for u in github_urls],
            qualification_score=score,
            signals=signals,
            repo_owner=owner,
            repo_name=repo_name,
            primary_url=primary_url,
            expand_text=(row['expand'] or '')[:300],
            connect_text=(row['connect'] or '')[:300],
        ))

    # Sort by qualification score descending
    qualified.sort(key=lambda a: a.qualification_score, reverse=True)

    # Cache results
    QUALIFY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {
        'timestamp': __import__('time').time(),
        'total_atoms': len(rows),
        'qualified_count': len(qualified),
        'atoms': [asdict(a) for a in qualified],
    }
    QUALIFY_CACHE.write_text(json.dumps(cache_data, indent=2))

    db.close()
    return qualified[:limit]


def display_qualified(atoms: List[QualifiedAtom]):
    """Display qualified atoms in a readable format."""
    print(f"\n{'='*70}")
    print(f"  WeEvolve INTEGRATE: QUALIFY ({len(atoms)} candidates)")
    print(f"{'='*70}\n")

    for i, atom in enumerate(atoms, 1):
        signal_str = ', '.join(f"{k}:{v}" for k, v in atom.signals.items())
        print(f"  {i:>2}. [{atom.qualification_score:.2f}] {atom.title[:55]}")
        print(f"      Repo: {atom.repo_owner}/{atom.repo_name}")
        print(f"      URL:  {atom.primary_url}")
        print(f"      Quality: {atom.quality:.2f} | Signals: {signal_str}")
        if atom.expand_text:
            print(f"      Expand: {atom.expand_text[:80]}...")
        print()

    print(f"{'='*70}")
    print(f"  Next: python3 -m weevolve.explore <repo_url>")
    print(f"{'='*70}\n")


def main():
    min_score = 0.3
    limit = 50
    output_json = False

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--min-score' and i + 1 < len(args):
            min_score = float(args[i + 1])
            i += 2
        elif args[i] == '--limit' and i + 1 < len(args):
            limit = int(args[i + 1])
            i += 2
        elif args[i] == '--json':
            output_json = True
            i += 1
        elif args[i] == '--no-cache':
            # Force refresh
            if QUALIFY_CACHE.exists():
                QUALIFY_CACHE.unlink()
            i += 1
        else:
            i += 1

    atoms = qualify_atoms(min_score=min_score, limit=limit)

    if output_json:
        print(json.dumps([asdict(a) for a in atoms], indent=2))
    else:
        display_qualified(atoms)


if __name__ == '__main__':
    main()
