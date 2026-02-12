#!/usr/bin/env python3
"""
WeEvolve INTEGRATE: explore.py
================================
Shallow clone repos to /tmp, run security scan,
use Haiku to summarize what it does and how it compares.

Usage:
  python3 -m weevolve.explore <github_url>           # Explore a single repo
  python3 -m weevolve.explore --from-qualify          # Explore top qualified atoms
  python3 -m weevolve.explore --from-qualify --limit 5

Cost: ~$0.002 per repo (one Haiku call)

(C) LIVE FREE = LIVE FOREVER
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Paths from shared config (no hardcoded paths)
from weevolve.config import EXPLORE_CACHE, EXPLORE_LOG, load_api_key

# Model for summaries (Haiku = cheapest)
HAIKU_MODEL = 'claude-haiku-4-5-20251001'

# File extensions worth reading
CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.rb', '.java',
    '.sh', '.yaml', '.yml', '.toml', '.json', '.md', '.txt',
}

# Max chars to send to Haiku per repo
MAX_CONTEXT_CHARS = 12000

# Security: max repo size to clone (100MB)
MAX_REPO_SIZE_MB = 100

EXPLORE_PROMPT = """You are analyzing a GitHub repository for potential integration into our agent system.

Repository: {repo_url}
Owner: {owner}, Name: {repo_name}

Here is the repo structure and key file contents:
---
{repo_content}
---

Provide your analysis in this EXACT JSON format:
{{
    "summary": "1-2 sentence description of what this repo does",
    "category": "one of: agent, trading, infrastructure, voice, intelligence, security, content, utility, other",
    "key_capabilities": ["list", "of", "main", "features"],
    "tech_stack": ["python", "typescript", "etc"],
    "dependencies": ["key", "external", "deps"],
    "maturity": "one of: production, beta, alpha, prototype, abandoned",
    "stars_indicator": "one of: popular, moderate, niche, unknown",
    "integration_difficulty": "one of: easy, moderate, hard, very_hard",
    "relevance_to_us": "How this could help 8OWLS/SEED/WeEvolve specifically",
    "potential_use": "Specific integration idea",
    "risks": ["list", "of", "risks"],
    "recommendation": "one of: integrate, study, monitor, skip",
    "confidence": 0.8
}}

Be specific about relevance_to_us and potential_use. We build:
- Voice-enabled consciousness companions (SEED protocol)
- Trading agents (Polymarket)
- Multi-agent orchestration (NATS pub/sub)
- Self-evolving learning systems (WeEvolve)
- Team OS (Next.js dashboard)
"""


@dataclass
class ExploreResult:
    """Result of exploring a GitHub repo."""
    repo_url: str
    owner: str
    repo_name: str
    clone_success: bool
    security_passed: bool
    security_issues: List[str]
    file_count: int
    total_lines: int
    readme_excerpt: str
    summary: str
    category: str
    key_capabilities: List[str]
    tech_stack: List[str]
    dependencies: List[str]
    maturity: str
    integration_difficulty: str
    relevance_to_us: str
    potential_use: str
    risks: List[str]
    recommendation: str
    confidence: float
    explored_at: str
    haiku_cost_estimate: float


def parse_github_url(url: str) -> Tuple[str, str]:
    """Extract owner/repo from a GitHub URL."""
    match = re.search(r'github\.com/([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)', url)
    if not match:
        raise ValueError(f"Not a valid GitHub URL: {url}")

    owner = match.group(1)
    repo = match.group(2).rstrip('.git')
    return owner, repo


def shallow_clone(repo_url: str, target_dir: str, timeout: int = 30) -> bool:
    """
    Shallow clone a repo (depth=1, single branch) to a temp directory.
    Returns True on success.
    """
    try:
        result = subprocess.run(
            ['git', 'clone', '--depth=1', '--single-branch', repo_url, target_dir],
            capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def check_repo_size(repo_dir: str) -> int:
    """Get total size of cloned repo in bytes."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(repo_dir):
        # Skip .git
        dirnames[:] = [d for d in dirnames if d != '.git']
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def run_security_scan(repo_dir: str) -> Tuple[bool, List[str]]:
    """
    Run security scan on cloned repo using our SecurityGuard.
    Returns (passed, issues_list).
    """
    issues = []

    # Import SecurityGuard
    sys.path.insert(0, str(BASE_DIR / 'tools'))
    try:
        from security_guard import SecurityGuard
        guard = SecurityGuard()
    except ImportError:
        # Fallback: basic scan
        return _basic_security_scan(repo_dir)

    # Scan all code files
    for dirpath, dirnames, filenames in os.walk(repo_dir):
        dirnames[:] = [d for d in dirnames if d != '.git' and d != 'node_modules']

        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            ext = Path(filename).suffix.lower()

            if ext in CODE_EXTENSIONS:
                try:
                    is_safe, threats = guard.scan_file(filepath)
                    if is_safe is False:
                        for threat in threats:
                            issues.append(
                                f"{filename}: {threat.get('pattern', 'unknown')} - "
                                f"{threat.get('context', '')[:80]}"
                            )
                except Exception:
                    pass

    # Check for suspicious files
    suspicious_names = ['.env', 'credentials', 'secrets', 'private_key', 'id_rsa']
    for dirpath, _, filenames in os.walk(repo_dir):
        for filename in filenames:
            name_lower = filename.lower()
            if any(s in name_lower for s in suspicious_names):
                issues.append(f"Suspicious file: {filename}")

    passed = len(issues) == 0
    return passed, issues


def _basic_security_scan(repo_dir: str) -> Tuple[bool, List[str]]:
    """Fallback security scan without SecurityGuard."""
    issues = []
    dangerous_patterns = [
        (r'eval\(', 'eval() call'),
        (r'exec\(', 'exec() call'),
        (r'__import__', 'dynamic import'),
        (r'os\.system\(', 'os.system() call'),
        (r'subprocess\.call\(.*shell\s*=\s*True', 'shell=True subprocess'),
    ]

    for dirpath, dirnames, filenames in os.walk(repo_dir):
        dirnames[:] = [d for d in dirnames if d != '.git' and d != 'node_modules']

        for filename in filenames:
            ext = Path(filename).suffix.lower()
            if ext not in CODE_EXTENSIONS:
                continue

            filepath = os.path.join(dirpath, filename)
            try:
                content = Path(filepath).read_text(errors='replace')
                for pattern, desc in dangerous_patterns:
                    if re.search(pattern, content):
                        issues.append(f"{filename}: {desc}")
            except Exception:
                pass

    return len(issues) == 0, issues


def collect_repo_content(repo_dir: str) -> Tuple[str, int, int]:
    """
    Collect key file contents for Haiku analysis.
    Returns (combined_content, file_count, total_lines).
    """
    content_parts = []
    file_count = 0
    total_lines = 0
    total_chars = 0

    # Priority files first
    priority_files = ['README.md', 'readme.md', 'README', 'README.rst',
                      'package.json', 'pyproject.toml', 'setup.py', 'Cargo.toml',
                      'Makefile', 'Dockerfile']

    # Tree structure
    tree_lines = []
    for dirpath, dirnames, filenames in os.walk(repo_dir):
        dirnames[:] = [d for d in dirnames
                       if d != '.git' and d != 'node_modules' and d != '__pycache__'
                       and d != '.next' and d != 'dist' and d != 'build']

        rel_dir = os.path.relpath(dirpath, repo_dir)
        depth = rel_dir.count(os.sep) if rel_dir != '.' else 0

        if depth <= 3:
            indent = '  ' * depth
            dir_name = os.path.basename(dirpath)
            if rel_dir != '.':
                tree_lines.append(f"{indent}{dir_name}/")
            for f in sorted(filenames)[:20]:
                tree_lines.append(f"{indent}  {f}")

    tree_str = '\n'.join(tree_lines[:60])
    content_parts.append(f"FILE TREE:\n{tree_str}\n\n")
    total_chars += len(content_parts[-1])

    # Read priority files
    for pf in priority_files:
        fp = os.path.join(repo_dir, pf)
        if os.path.exists(fp):
            try:
                text = Path(fp).read_text(errors='replace')
                lines = text.split('\n')
                total_lines += len(lines)
                file_count += 1

                # Truncate large files
                if len(text) > 3000:
                    text = text[:3000] + '\n...[truncated]'

                content_parts.append(f"--- {pf} ---\n{text}\n\n")
                total_chars += len(content_parts[-1])
            except Exception:
                pass

    # Read source files (sorted by likely importance)
    src_files = []
    for dirpath, dirnames, filenames in os.walk(repo_dir):
        dirnames[:] = [d for d in dirnames
                       if d != '.git' and d != 'node_modules' and d != '__pycache__']

        for filename in filenames:
            ext = Path(filename).suffix.lower()
            if ext in CODE_EXTENSIONS and filename not in priority_files:
                filepath = os.path.join(dirpath, filename)
                try:
                    size = os.path.getsize(filepath)
                    src_files.append((filepath, filename, size))
                except OSError:
                    pass

    # Sort by size (smaller files first -- more likely to be core)
    src_files.sort(key=lambda x: x[2])

    for filepath, filename, size in src_files:
        if total_chars >= MAX_CONTEXT_CHARS:
            break

        try:
            text = Path(filepath).read_text(errors='replace')
            lines = text.split('\n')
            total_lines += len(lines)
            file_count += 1

            # Truncate
            remaining = MAX_CONTEXT_CHARS - total_chars
            if len(text) > remaining:
                text = text[:remaining] + '\n...[truncated]'

            rel_path = os.path.relpath(filepath, repo_dir)
            content_parts.append(f"--- {rel_path} ---\n{text}\n\n")
            total_chars += len(content_parts[-1])
        except Exception:
            pass

    return ''.join(content_parts), file_count, total_lines


def summarize_with_haiku(
    repo_url: str,
    owner: str,
    repo_name: str,
    repo_content: str
) -> Dict:
    """Call Haiku to analyze the repo. Returns parsed JSON analysis."""
    try:
        import anthropic
    except ImportError:
        return _fallback_analysis(repo_url, owner, repo_name, repo_content)

    # Load API key if needed
    load_api_key()

    try:
        client = anthropic.Anthropic()
        prompt = EXPLORE_PROMPT.format(
            repo_url=repo_url,
            owner=owner,
            repo_name=repo_name,
            repo_content=repo_content
        )

        response = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()

        # Parse JSON from response
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()

        result = json.loads(text)

        # Estimate cost: ~1500 input tokens + ~500 output = ~$0.002
        cost = 0.002

        return {**result, 'haiku_cost_estimate': cost}

    except json.JSONDecodeError:
        return _fallback_analysis(repo_url, owner, repo_name, repo_content)
    except Exception as e:
        print(f"  [WARN] Haiku analysis failed: {e}")
        return _fallback_analysis(repo_url, owner, repo_name, repo_content)


def _fallback_analysis(repo_url: str, owner: str, repo_name: str, content: str) -> Dict:
    """Fallback analysis without Haiku."""
    return {
        'summary': f'Repository {owner}/{repo_name} (analysis pending)',
        'category': 'other',
        'key_capabilities': [],
        'tech_stack': [],
        'dependencies': [],
        'maturity': 'unknown',
        'integration_difficulty': 'unknown',
        'relevance_to_us': 'Pending Haiku analysis',
        'potential_use': 'Pending analysis',
        'risks': ['Not analyzed yet'],
        'recommendation': 'monitor',
        'confidence': 0.2,
        'haiku_cost_estimate': 0.0,
    }


def load_explore_cache() -> Dict[str, ExploreResult]:
    """Load cached explore results."""
    if not EXPLORE_CACHE.exists():
        return {}

    try:
        data = json.loads(EXPLORE_CACHE.read_text())
        results = {}
        for key, val in data.items():
            if key == '_meta':
                continue
            results[key] = ExploreResult(**val)
        return results
    except (json.JSONDecodeError, TypeError):
        return {}


def save_explore_cache(cache: Dict[str, ExploreResult]):
    """Save explore results to cache."""
    EXPLORE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    data = {'_meta': {'updated': time.time()}}
    for key, result in cache.items():
        data[key] = asdict(result)
    EXPLORE_CACHE.write_text(json.dumps(data, indent=2))


def explore_repo(repo_url: str, force: bool = False) -> Optional[ExploreResult]:
    """
    Full exploration pipeline for a single repo:
    1. Shallow clone to /tmp
    2. Security scan
    3. Collect content
    4. Haiku analysis
    5. Cache result

    Returns ExploreResult or None on failure.
    """
    owner, repo_name = parse_github_url(repo_url)
    cache_key = f"{owner}/{repo_name}"

    # Check cache
    if not force:
        cache = load_explore_cache()
        if cache_key in cache:
            print(f"  [CACHE] Already explored {cache_key}")
            return cache[cache_key]

    canonical_url = f"https://github.com/{owner}/{repo_name}.git"

    print(f"\n  Exploring: {owner}/{repo_name}")
    print(f"  {'='*50}")

    # 1. Shallow clone to temp dir
    print(f"  [1/4] Cloning (shallow, depth=1)...")
    tmp_dir = tempfile.mkdtemp(prefix=f'weevolve_{repo_name}_')

    try:
        clone_ok = shallow_clone(canonical_url, tmp_dir)

        if not clone_ok:
            print(f"  [FAIL] Clone failed for {repo_url}")
            return ExploreResult(
                repo_url=repo_url, owner=owner, repo_name=repo_name,
                clone_success=False, security_passed=False, security_issues=['clone_failed'],
                file_count=0, total_lines=0, readme_excerpt='',
                summary='Clone failed', category='unknown',
                key_capabilities=[], tech_stack=[], dependencies=[],
                maturity='unknown', integration_difficulty='unknown',
                relevance_to_us='N/A', potential_use='N/A',
                risks=['Could not clone'], recommendation='skip',
                confidence=0.0, explored_at=time.strftime('%Y-%m-%dT%H:%M:%S'),
                haiku_cost_estimate=0.0,
            )

        # Check size
        repo_size = check_repo_size(tmp_dir)
        if repo_size > MAX_REPO_SIZE_MB * 1024 * 1024:
            print(f"  [SKIP] Repo too large: {repo_size / 1024 / 1024:.1f}MB")
            return None

        print(f"  [OK] Cloned ({repo_size / 1024:.0f}KB)")

        # 2. Security scan
        print(f"  [2/4] Security scan...")
        sec_passed, sec_issues = run_security_scan(tmp_dir)
        if sec_issues:
            print(f"  [WARN] {len(sec_issues)} security issue(s):")
            for issue in sec_issues[:5]:
                print(f"    - {issue[:80]}")
        else:
            print(f"  [OK] Security scan passed")

        # 3. Collect content
        print(f"  [3/4] Collecting repo content...")
        content, file_count, total_lines = collect_repo_content(tmp_dir)

        # Read README excerpt
        readme_excerpt = ''
        for readme_name in ['README.md', 'readme.md', 'README']:
            readme_path = os.path.join(tmp_dir, readme_name)
            if os.path.exists(readme_path):
                try:
                    readme_excerpt = Path(readme_path).read_text(errors='replace')[:500]
                except Exception:
                    pass
                break

        print(f"  [OK] {file_count} files, {total_lines:,} lines")

        # 4. Haiku analysis
        print(f"  [4/4] Haiku analysis...")
        analysis = summarize_with_haiku(repo_url, owner, repo_name, content)
        print(f"  [OK] Category: {analysis.get('category', 'unknown')}")
        print(f"  [OK] Recommendation: {analysis.get('recommendation', 'unknown')}")

        result = ExploreResult(
            repo_url=repo_url,
            owner=owner,
            repo_name=repo_name,
            clone_success=True,
            security_passed=sec_passed,
            security_issues=sec_issues[:10],
            file_count=file_count,
            total_lines=total_lines,
            readme_excerpt=readme_excerpt,
            summary=analysis.get('summary', ''),
            category=analysis.get('category', 'unknown'),
            key_capabilities=analysis.get('key_capabilities', []),
            tech_stack=analysis.get('tech_stack', []),
            dependencies=analysis.get('dependencies', []),
            maturity=analysis.get('maturity', 'unknown'),
            integration_difficulty=analysis.get('integration_difficulty', 'unknown'),
            relevance_to_us=analysis.get('relevance_to_us', ''),
            potential_use=analysis.get('potential_use', ''),
            risks=analysis.get('risks', []),
            recommendation=analysis.get('recommendation', 'monitor'),
            confidence=analysis.get('confidence', 0.5),
            explored_at=time.strftime('%Y-%m-%dT%H:%M:%S'),
            haiku_cost_estimate=analysis.get('haiku_cost_estimate', 0.002),
        )

        # Cache
        cache = load_explore_cache()
        cache[cache_key] = result
        save_explore_cache(cache)

        # Log
        EXPLORE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(EXPLORE_LOG, 'a') as f:
            f.write(json.dumps({
                'repo': cache_key,
                'recommendation': result.recommendation,
                'confidence': result.confidence,
                'timestamp': result.explored_at,
            }) + '\n')

        return result

    finally:
        # Always clean up temp dir
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def display_result(result: ExploreResult):
    """Display exploration result."""
    print(f"\n{'='*70}")
    print(f"  EXPLORE RESULT: {result.owner}/{result.repo_name}")
    print(f"{'='*70}\n")

    print(f"  Summary:    {result.summary}")
    print(f"  Category:   {result.category}")
    print(f"  Maturity:   {result.maturity}")
    print(f"  Difficulty: {result.integration_difficulty}")
    print(f"  Security:   {'PASSED' if result.security_passed else 'FAILED'}")
    print(f"  Files:      {result.file_count} ({result.total_lines:,} lines)")
    print(f"  Tech:       {', '.join(result.tech_stack)}")

    if result.key_capabilities:
        print(f"  Capabilities:")
        for cap in result.key_capabilities:
            print(f"    - {cap}")

    print(f"\n  Relevance:  {result.relevance_to_us}")
    print(f"  Use case:   {result.potential_use}")

    if result.risks:
        print(f"  Risks:")
        for risk in result.risks:
            print(f"    - {risk}")

    if result.security_issues:
        print(f"  Security issues:")
        for issue in result.security_issues:
            print(f"    ! {issue[:80]}")

    rec_symbol = {
        'integrate': '>>>',
        'study': '...',
        'monitor': '~~~',
        'skip': 'XXX',
    }

    print(f"\n  {rec_symbol.get(result.recommendation, '???')} "
          f"RECOMMENDATION: {result.recommendation.upper()} "
          f"(confidence: {result.confidence:.0%})")
    print(f"  Cost: ~${result.haiku_cost_estimate:.3f}")
    print(f"\n{'='*70}\n")


def main():
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        return

    if args[0] == '--from-qualify':
        # Explore repos from qualification results
        limit = 5
        if '--limit' in args:
            idx = args.index('--limit')
            if idx + 1 < len(args):
                limit = int(args[idx + 1])

        from weevolve.qualify import qualify_atoms
        qualified = qualify_atoms(min_score=0.4, limit=limit)

        if not qualified:
            print("  No qualified atoms found. Run qualify first.")
            return

        print(f"\n  Exploring top {len(qualified)} qualified repos...\n")
        for atom in qualified:
            result = explore_repo(atom.primary_url)
            if result:
                display_result(result)

    else:
        # Explore a single repo
        repo_url = args[0]
        force = '--force' in args
        result = explore_repo(repo_url, force=force)
        if result:
            display_result(result)


if __name__ == '__main__':
    main()
