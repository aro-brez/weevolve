#!/usr/bin/env python3
"""
WeEvolve Project Evolution - Attach evolution to any project
=============================================================
Scan any project directory, find improvements, suggest upgrades,
auto-apply approved changes. Works for Python, Node, Rust, Go, anything.

Usage:
  weevolve project                   # Scan current directory
  weevolve project --apply           # Scan + auto-apply approved improvements
  weevolve project --daemon          # Run recurring evolution checks
  weevolve project --competitive     # Include competitive GitHub analysis
  weevolve project --path /some/dir  # Scan a specific directory

SEED phases applied to projects:
  PERCEIVE  -> Scan files, deps, structure, test coverage
  CONNECT   -> Find similar projects, compare patterns
  LEARN     -> Extract actionable improvements
  QUESTION  -> Challenge architecture decisions
  EXPAND    -> Suggest new capabilities
  SHARE     -> Share improvements across projects
  RECEIVE   -> Accept community best practices
  IMPROVE   -> Meta-learn what makes projects better

(C) LIVE FREE = LIVE FOREVER
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field as datafield
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from weevolve.config import DATA_DIR, load_api_key

# ANSI colors
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

SEED_PHASES = [
    ("LYRA", "PERCEIVE", CYAN),
    ("PRISM", "CONNECT", MAGENTA),
    ("SAGE", "LEARN", GREEN),
    ("QUEST", "QUESTION", YELLOW),
    ("NOVA", "EXPAND", BLUE),
    ("ECHO", "SHARE", LIME),
    ("LUNA", "RECEIVE", DIM),
    ("SOWL", "IMPROVE", RED),
]

# Model for analysis (Haiku for cost efficiency)
ANALYSIS_MODEL = "claude-haiku-4-5-20251001"

# Project evolution state directory
PROJECT_EVOLUTION_DIR = DATA_DIR / "project_evolution"

# Manifest files by ecosystem
MANIFEST_FILES = {
    "python": ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile"],
    "node": ["package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml"],
    "rust": ["Cargo.toml", "Cargo.lock"],
    "go": ["go.mod", "go.sum"],
    "ruby": ["Gemfile", "Gemfile.lock"],
    "java": ["pom.xml", "build.gradle", "build.gradle.kts"],
    "swift": ["Package.swift"],
    "dotnet": ["*.csproj", "*.fsproj", "*.sln"],
}

# Common project files that indicate project characteristics
PROJECT_INDICATORS = {
    "has_tests": ["tests/", "test/", "spec/", "__tests__/", "pytest.ini", "jest.config.*"],
    "has_ci": [".github/workflows/", ".gitlab-ci.yml", "Jenkinsfile", ".circleci/"],
    "has_linting": [".eslintrc*", ".flake8", "ruff.toml", ".pylintrc", "biome.json"],
    "has_formatting": [".prettierrc*", "pyproject.toml", ".editorconfig"],
    "has_docker": ["Dockerfile", "docker-compose.yml", "docker-compose.yaml"],
    "has_docs": ["docs/", "doc/", "README.md", "CONTRIBUTING.md"],
    "has_security": [".env.example", "SECURITY.md", ".snyk"],
    "has_typing": ["tsconfig.json", "py.typed", "mypy.ini"],
}

# Improvement analysis prompt
PROJECT_ANALYSIS_PROMPT = """You are analyzing a software project through the SEED protocol to find improvements.

Project scan results:
---
{scan_json}
---

Analyze this project and suggest improvements. Be SPECIFIC and ACTIONABLE.
For each improvement, provide a confidence score (0.0-1.0) based on how certain you are
this would help, and how easy it is to implement.

Respond in this EXACT JSON format:
{{
    "project_summary": "1-2 sentence description of what this project is",
    "tech_stack": ["language", "framework", "etc"],
    "health_score": 0.7,
    "improvements": [
        {{
            "title": "Short title of the improvement",
            "category": "one of: tests, dependencies, security, ci, docs, performance, patterns, architecture",
            "description": "What to do and why",
            "confidence": 0.85,
            "effort": "one of: trivial, easy, moderate, hard",
            "impact": "one of: low, medium, high, critical",
            "auto_applicable": true,
            "command": "optional shell command to apply this (e.g., 'pip install --upgrade fastapi')",
            "files_affected": ["list of files to create or modify"]
        }}
    ],
    "strengths": ["what this project already does well"],
    "competitive_gap": "what similar projects do that this one doesn't"
}}

Scoring:
- 0.9-1.0: Must fix immediately (security, broken deps)
- 0.7-0.89: Strong recommendation (missing tests, outdated deps)
- 0.5-0.69: Good to have (better patterns, docs improvements)
- 0.3-0.49: Nice to have (style, minor optimizations)

Rules:
- Maximum 10 improvements, sorted by confidence DESC
- Every improvement must be SPECIFIC (not "add tests" but "add pytest with conftest.py covering the 3 main modules")
- Skip trivial style issues unless there are no bigger improvements
- If the project is already well-maintained, say so and suggest fewer improvements
- Be size-appropriate: don't suggest Kubernetes for a 100-line script
"""


@dataclass
class ProjectScan:
    """Results of scanning a project directory."""
    path: str
    name: str
    ecosystems: List[str]
    file_count: int
    total_lines: int
    manifests_found: Dict[str, str]
    indicators: Dict[str, bool]
    top_level_files: List[str]
    source_files: Dict[str, int]  # extension -> count
    readme_excerpt: str
    claude_md_excerpt: str
    dependency_count: int
    dependencies: List[str]
    has_git: bool
    git_remotes: List[str]
    last_commit_date: str
    scanned_at: str


@dataclass
class ProjectImprovement:
    """A single suggested improvement."""
    title: str
    category: str
    description: str
    confidence: float
    effort: str
    impact: str
    auto_applicable: bool
    command: str
    files_affected: List[str]
    applied: bool = False
    applied_at: str = ""


@dataclass
class ProjectEvolutionState:
    """Evolution state for a specific project."""
    project_path: str
    project_name: str
    scan_count: int
    last_scan: str
    health_score: float
    improvements_suggested: int
    improvements_applied: int
    tech_stack: List[str]
    history: List[Dict[str, Any]]


def _phase_log(idx: int, detail: str = "") -> None:
    """Print a colored SEED phase indicator."""
    if 0 <= idx < len(SEED_PHASES):
        owl, phase, color = SEED_PHASES[idx]
        print(f"  {color}{owl}{RESET} {DIM}{phase}{RESET} {detail}")


def _read_file_safe(path: Path, max_chars: int = 3000) -> str:
    """Read a file safely, truncating if too large."""
    try:
        text = path.read_text(errors="replace")
        if len(text) > max_chars:
            return text[:max_chars] + "\n...[truncated]"
        return text
    except Exception:
        return ""


def _count_lines(path: Path) -> int:
    """Count lines in a file."""
    try:
        return len(path.read_text(errors="replace").splitlines())
    except Exception:
        return 0


def _run_cmd(cmd: List[str], cwd: str, timeout: int = 10) -> Tuple[bool, str]:
    """Run a shell command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd
        )
        return result.returncode == 0, result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False, ""


def _parse_dependencies_from_requirements(path: Path) -> List[str]:
    """Extract dependency names from requirements.txt."""
    deps = []
    try:
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            # Extract package name (before ==, >=, ~=, etc.)
            name = re.split(r"[>=<~!;\[]", line)[0].strip()
            if name:
                deps.append(name)
    except Exception:
        pass
    return deps


def _parse_dependencies_from_package_json(path: Path) -> List[str]:
    """Extract dependency names from package.json."""
    deps = []
    try:
        data = json.loads(path.read_text())
        for section in ["dependencies", "devDependencies", "peerDependencies"]:
            if section in data:
                deps.extend(data[section].keys())
    except Exception:
        pass
    return deps


def _parse_dependencies_from_pyproject(path: Path) -> List[str]:
    """Extract dependency names from pyproject.toml (simple parser)."""
    deps = []
    try:
        content = path.read_text(errors="replace")
        # Simple regex for dependencies = [...] block
        dep_match = re.search(
            r'dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL
        )
        if dep_match:
            block = dep_match.group(1)
            for line in block.splitlines():
                line = line.strip().strip(",").strip('"').strip("'")
                if line:
                    name = re.split(r"[>=<~!;\[]", line)[0].strip()
                    if name:
                        deps.append(name)
    except Exception:
        pass
    return deps


def _parse_dependencies_from_cargo(path: Path) -> List[str]:
    """Extract dependency names from Cargo.toml."""
    deps = []
    try:
        content = path.read_text(errors="replace")
        in_deps = False
        for line in content.splitlines():
            if re.match(r'\[dependencies\]', line) or re.match(r'\[dev-dependencies\]', line):
                in_deps = True
                continue
            if line.startswith("[") and in_deps:
                in_deps = False
                continue
            if in_deps and "=" in line:
                name = line.split("=")[0].strip()
                if name:
                    deps.append(name)
    except Exception:
        pass
    return deps


def _parse_dependencies_from_gomod(path: Path) -> List[str]:
    """Extract dependency paths from go.mod."""
    deps = []
    try:
        content = path.read_text(errors="replace")
        in_require = False
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("require ("):
                in_require = True
                continue
            if line == ")" and in_require:
                in_require = False
                continue
            if in_require and line:
                parts = line.split()
                if parts:
                    deps.append(parts[0])
            elif line.startswith("require ") and "(" not in line:
                parts = line.split()
                if len(parts) >= 2:
                    deps.append(parts[1])
    except Exception:
        pass
    return deps


# ============================================================================
# PHASE 1: PERCEIVE -- Scan the project
# ============================================================================

def scan_project(project_path: Optional[str] = None) -> ProjectScan:
    """
    Deep scan of a project directory. Understands any ecosystem.
    Returns a ProjectScan dataclass with all discovered information.
    """
    base = Path(project_path) if project_path else Path.cwd()
    base = base.resolve()

    _phase_log(0, f"scanning {base.name}...")

    # Detect ecosystems
    ecosystems = []
    manifests_found = {}

    for ecosystem, manifest_list in MANIFEST_FILES.items():
        for manifest in manifest_list:
            if "*" in manifest:
                matches = list(base.glob(manifest))
                if matches:
                    ecosystems.append(ecosystem)
                    manifests_found[ecosystem] = matches[0].name
                    break
            elif (base / manifest).exists():
                ecosystems.append(ecosystem)
                manifests_found[ecosystem] = manifest
                break

    # Detect indicators
    indicators = {}
    for indicator_name, patterns in PROJECT_INDICATORS.items():
        found = False
        for pattern in patterns:
            if "*" in pattern:
                if list(base.glob(pattern)):
                    found = True
                    break
            elif (base / pattern).exists():
                found = True
                break
        indicators[indicator_name] = found

    # Count source files by extension
    source_files = {}
    file_count = 0
    total_lines = 0
    skip_dirs = {
        ".git", "node_modules", "__pycache__", ".next", "dist", "build",
        "target", ".tox", ".venv", "venv", "env", ".mypy_cache",
        ".pytest_cache", "coverage", ".turbo",
    }

    for dirpath, dirnames, filenames in os.walk(str(base)):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for filename in filenames:
            filepath = Path(dirpath) / filename
            ext = filepath.suffix.lower()
            if ext in {
                ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".rb",
                ".java", ".swift", ".kt", ".c", ".cpp", ".h", ".cs", ".fs",
                ".sh", ".yaml", ".yml", ".toml", ".json", ".md",
            }:
                source_files[ext] = source_files.get(ext, 0) + 1
                file_count += 1
                total_lines += _count_lines(filepath)

    # Read top-level files
    try:
        top_level = sorted(
            p.name for p in base.iterdir()
            if not p.name.startswith(".")
        )[:30]
    except Exception:
        top_level = []

    # Read README
    readme_excerpt = ""
    for readme_name in ["README.md", "readme.md", "README.rst", "README"]:
        readme_path = base / readme_name
        if readme_path.exists():
            readme_excerpt = _read_file_safe(readme_path, 2000)
            break

    # Read CLAUDE.md
    claude_md_excerpt = ""
    claude_md_path = base / "CLAUDE.md"
    if claude_md_path.exists():
        claude_md_excerpt = _read_file_safe(claude_md_path, 1500)

    # Parse dependencies
    dependencies = []
    if (base / "requirements.txt").exists():
        dependencies = _parse_dependencies_from_requirements(base / "requirements.txt")
    elif (base / "pyproject.toml").exists():
        dependencies = _parse_dependencies_from_pyproject(base / "pyproject.toml")
    if (base / "package.json").exists():
        dependencies.extend(_parse_dependencies_from_package_json(base / "package.json"))
    if (base / "Cargo.toml").exists():
        dependencies.extend(_parse_dependencies_from_cargo(base / "Cargo.toml"))
    if (base / "go.mod").exists():
        dependencies.extend(_parse_dependencies_from_gomod(base / "go.mod"))

    # Git info
    has_git = (base / ".git").exists()
    git_remotes = []
    last_commit_date = ""

    if has_git:
        ok, remotes_out = _run_cmd(["git", "remote", "-v"], str(base))
        if ok and remotes_out:
            for line in remotes_out.splitlines():
                parts = line.split()
                if len(parts) >= 2 and "(fetch)" in line:
                    git_remotes.append(parts[1])

        ok, date_out = _run_cmd(
            ["git", "log", "-1", "--format=%ci"], str(base)
        )
        if ok:
            last_commit_date = date_out

    return ProjectScan(
        path=str(base),
        name=base.name,
        ecosystems=ecosystems,
        file_count=file_count,
        total_lines=total_lines,
        manifests_found=manifests_found,
        indicators=indicators,
        top_level_files=top_level,
        source_files=source_files,
        readme_excerpt=readme_excerpt,
        claude_md_excerpt=claude_md_excerpt,
        dependency_count=len(dependencies),
        dependencies=dependencies[:50],  # cap for prompt size
        has_git=has_git,
        git_remotes=git_remotes,
        last_commit_date=last_commit_date,
        scanned_at=datetime.now(timezone.utc).isoformat(),
    )


# ============================================================================
# PHASE 2: CONNECT -- Find similar projects (competitive analysis)
# ============================================================================

def find_competitors(scan: ProjectScan, limit: int = 3) -> List[Dict]:
    """
    Find similar projects on GitHub using the explore module.
    Returns list of competitor summaries.
    """
    _phase_log(1, "finding similar projects...")

    # Build search terms from project characteristics
    search_terms = []
    if scan.readme_excerpt:
        # Extract first meaningful line from README
        for line in scan.readme_excerpt.splitlines():
            line = line.strip().lstrip("#").strip()
            if len(line) > 10 and not line.startswith("!") and not line.startswith("["):
                search_terms.append(line[:80])
                break

    if scan.ecosystems:
        search_terms.extend(scan.ecosystems)

    if not search_terms:
        search_terms.append(scan.name)

    # For now, return a placeholder -- competitive analysis via GitHub API
    # would need a search endpoint. The explore module handles individual repos.
    # This can be enhanced with GitHub search API or web search later.
    _phase_log(1, f"search terms: {', '.join(search_terms[:3])}")

    return []


# ============================================================================
# PHASE 3: LEARN -- Generate improvements using Claude
# ============================================================================

def generate_improvements(scan: ProjectScan, competitors: List[Dict] = None) -> Dict:
    """
    Analyze the project scan and generate scored improvements.
    Uses Claude Haiku for cost-efficient analysis (~$0.003 per call).
    """
    _phase_log(2, "extracting improvements...")

    # Build scan JSON for the prompt (exclude very large fields)
    scan_data = {
        "name": scan.name,
        "path": scan.path,
        "ecosystems": scan.ecosystems,
        "file_count": scan.file_count,
        "total_lines": scan.total_lines,
        "manifests": scan.manifests_found,
        "indicators": scan.indicators,
        "top_level_files": scan.top_level_files,
        "source_files": scan.source_files,
        "dependency_count": scan.dependency_count,
        "dependencies": scan.dependencies[:30],
        "has_git": scan.has_git,
        "last_commit": scan.last_commit_date,
        "readme_first_500": scan.readme_excerpt[:500],
    }

    if competitors:
        scan_data["competitors"] = competitors[:3]

    scan_json = json.dumps(scan_data, indent=2)

    # Try Claude analysis
    try:
        import anthropic
        load_api_key()

        client = anthropic.Anthropic()
        prompt = PROJECT_ANALYSIS_PROMPT.format(scan_json=scan_json)

        response = client.messages.create(
            model=ANALYSIS_MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()

        # Extract JSON from response
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        result = json.loads(text)
        return result

    except ImportError:
        return _fallback_improvements(scan)
    except json.JSONDecodeError:
        return _fallback_improvements(scan)
    except Exception as exc:
        print(f"  {RED}[WARN]{RESET} Claude analysis failed: {exc}")
        return _fallback_improvements(scan)


def _fallback_improvements(scan: ProjectScan) -> Dict:
    """Generate improvements using heuristics when Claude is unavailable."""
    improvements = []

    # Check for missing tests
    if not scan.indicators.get("has_tests"):
        test_framework = "pytest" if "python" in scan.ecosystems else "jest"
        if "rust" in scan.ecosystems:
            test_framework = "cargo test (built-in)"
        elif "go" in scan.ecosystems:
            test_framework = "go test (built-in)"

        improvements.append({
            "title": f"Add test suite ({test_framework})",
            "category": "tests",
            "description": f"No tests found. Add {test_framework} with tests for core modules.",
            "confidence": 0.95,
            "effort": "moderate",
            "impact": "high",
            "auto_applicable": False,
            "command": "",
            "files_affected": ["tests/"],
        })

    # Check for missing CI
    if not scan.indicators.get("has_ci") and scan.has_git:
        improvements.append({
            "title": "Add CI/CD pipeline",
            "category": "ci",
            "description": "No CI configuration found. Add GitHub Actions workflow for testing and linting.",
            "confidence": 0.90,
            "effort": "easy",
            "impact": "high",
            "auto_applicable": True,
            "command": "",
            "files_affected": [".github/workflows/ci.yml"],
        })

    # Check for missing linting
    if not scan.indicators.get("has_linting"):
        linter = "ruff" if "python" in scan.ecosystems else "eslint"
        if "rust" in scan.ecosystems:
            linter = "clippy (built-in)"

        improvements.append({
            "title": f"Add linting ({linter})",
            "category": "patterns",
            "description": f"No linter configured. Add {linter} for consistent code quality.",
            "confidence": 0.85,
            "effort": "easy",
            "impact": "medium",
            "auto_applicable": True,
            "command": "",
            "files_affected": [],
        })

    # Check for missing docs
    if not scan.readme_excerpt:
        improvements.append({
            "title": "Add README.md",
            "category": "docs",
            "description": "No README found. Every project needs a README with setup instructions.",
            "confidence": 0.90,
            "effort": "easy",
            "impact": "medium",
            "auto_applicable": False,
            "command": "",
            "files_affected": ["README.md"],
        })

    # Check for missing security patterns
    if not scan.indicators.get("has_security"):
        improvements.append({
            "title": "Add .env.example and security docs",
            "category": "security",
            "description": "No .env.example found. Document required environment variables.",
            "confidence": 0.75,
            "effort": "trivial",
            "impact": "medium",
            "auto_applicable": True,
            "command": "",
            "files_affected": [".env.example"],
        })

    # Check for missing Docker
    if not scan.indicators.get("has_docker") and scan.file_count > 10:
        improvements.append({
            "title": "Add Dockerfile for containerization",
            "category": "architecture",
            "description": "No Docker configuration. Add a Dockerfile for reproducible builds.",
            "confidence": 0.60,
            "effort": "moderate",
            "impact": "medium",
            "auto_applicable": False,
            "command": "",
            "files_affected": ["Dockerfile"],
        })

    return {
        "project_summary": f"{scan.name}: {', '.join(scan.ecosystems)} project with {scan.file_count} files",
        "tech_stack": scan.ecosystems,
        "health_score": _compute_health_score(scan),
        "improvements": improvements,
        "strengths": _detect_strengths(scan),
        "competitive_gap": "Run with --competitive flag for GitHub comparison",
    }


def _compute_health_score(scan: ProjectScan) -> float:
    """Compute a 0-1 health score from scan indicators."""
    score = 0.3  # Base score for existing
    checks = [
        ("has_tests", 0.15),
        ("has_ci", 0.10),
        ("has_linting", 0.10),
        ("has_formatting", 0.05),
        ("has_docs", 0.10),
        ("has_security", 0.10),
        ("has_typing", 0.05),
        ("has_docker", 0.05),
    ]
    for indicator, weight in checks:
        if scan.indicators.get(indicator):
            score += weight
    return round(min(1.0, score), 2)


def _detect_strengths(scan: ProjectScan) -> List[str]:
    """Detect what the project already does well."""
    strengths = []
    if scan.indicators.get("has_tests"):
        strengths.append("Has test suite")
    if scan.indicators.get("has_ci"):
        strengths.append("Has CI/CD pipeline")
    if scan.indicators.get("has_linting"):
        strengths.append("Has linting configured")
    if scan.indicators.get("has_typing"):
        strengths.append("Has type checking")
    if scan.indicators.get("has_docker"):
        strengths.append("Has Docker support")
    if scan.indicators.get("has_docs"):
        strengths.append("Has documentation")
    if scan.has_git:
        strengths.append("Version controlled with Git")
    if scan.dependency_count > 0:
        strengths.append(f"Manages {scan.dependency_count} dependencies")
    return strengths


# ============================================================================
# PHASE 4: QUESTION -- Validate and score improvements
# ============================================================================

def validate_improvements(improvements: List[Dict], scan: ProjectScan) -> List[ProjectImprovement]:
    """
    Validate and filter improvements. Apply size-appropriate checks.
    """
    _phase_log(3, "challenging assumptions...")

    validated = []
    for imp in improvements:
        confidence = imp.get("confidence", 0.5)

        # Size-appropriate filtering
        if scan.file_count < 5 and imp.get("category") in {"ci", "architecture"}:
            confidence *= 0.5  # Don't suggest enterprise patterns for tiny projects

        if scan.total_lines < 100 and imp.get("effort") == "hard":
            confidence *= 0.5  # Don't suggest hard improvements for tiny projects

        # Skip low-confidence items
        if confidence < 0.3:
            continue

        validated.append(ProjectImprovement(
            title=imp.get("title", "Unknown"),
            category=imp.get("category", "other"),
            description=imp.get("description", ""),
            confidence=round(confidence, 2),
            effort=imp.get("effort", "moderate"),
            impact=imp.get("impact", "medium"),
            auto_applicable=imp.get("auto_applicable", False),
            command=imp.get("command", ""),
            files_affected=imp.get("files_affected", []),
        ))

    # Sort by confidence descending
    validated.sort(key=lambda x: x.confidence, reverse=True)
    return validated[:10]  # Max 10 improvements


# ============================================================================
# PHASE 5-6: EXPAND + SHARE -- Display results
# ============================================================================

def display_results(
    scan: ProjectScan,
    analysis: Dict,
    improvements: List[ProjectImprovement],
) -> None:
    """Display the project evolution analysis."""
    health = analysis.get("health_score", 0.5)
    summary = analysis.get("project_summary", scan.name)
    strengths = analysis.get("strengths", [])
    tech = analysis.get("tech_stack", scan.ecosystems)

    # Header
    print(f"\n{'=' * 60}")
    print(f"  (*) WeEvolve PROJECT EVOLUTION")
    print(f"{'=' * 60}")
    print(f"  Project:  {scan.name}")
    print(f"  Path:     {scan.path}")
    print(f"  Summary:  {summary}")
    print(f"  Stack:    {', '.join(tech)}")
    print(f"  Files:    {scan.file_count} ({scan.total_lines:,} lines)")
    print(f"  Deps:     {scan.dependency_count}")

    # Health bar
    bar_len = int(health * 20)
    bar = chr(0x2588) * bar_len + chr(0x2591) * (20 - bar_len)
    if health >= 0.8:
        color = GREEN
    elif health >= 0.5:
        color = YELLOW
    else:
        color = RED
    print(f"  Health:   {color}{bar}{RESET} {health:.0%}")

    # Strengths
    if strengths:
        print(f"\n  {GREEN}Strengths:{RESET}")
        for s in strengths:
            print(f"    + {s}")

    # Improvements
    if improvements:
        _phase_log(4, f"found {len(improvements)} improvements")
        print(f"\n  {BOLD}Suggested Improvements:{RESET}")
        print()
        for i, imp in enumerate(improvements, 1):
            conf_color = GREEN if imp.confidence >= 0.8 else (YELLOW if imp.confidence >= 0.5 else DIM)
            effort_color = GREEN if imp.effort == "trivial" else (
                LIME if imp.effort == "easy" else (
                    YELLOW if imp.effort == "moderate" else RED
                )
            )
            impact_color = RED if imp.impact == "critical" else (
                YELLOW if imp.impact == "high" else (
                    LIME if imp.impact == "medium" else DIM
                )
            )

            print(f"  {BOLD}{i:>2}.{RESET} [{conf_color}{imp.confidence:.2f}{RESET}] {imp.title}")
            print(f"      {DIM}Category:{RESET} {imp.category}  "
                  f"{DIM}Effort:{RESET} {effort_color}{imp.effort}{RESET}  "
                  f"{DIM}Impact:{RESET} {impact_color}{imp.impact}{RESET}")
            print(f"      {DIM}{imp.description}{RESET}")
            if imp.command:
                print(f"      {CYAN}$ {imp.command}{RESET}")
            if imp.files_affected:
                print(f"      {DIM}Files: {', '.join(imp.files_affected)}{RESET}")
            print()
    else:
        print(f"\n  {GREEN}No improvements needed -- project looks healthy!{RESET}")

    # Competitive gap
    gap = analysis.get("competitive_gap", "")
    if gap:
        print(f"  {DIM}Competitive gap: {gap}{RESET}")

    print(f"{'=' * 60}")


# ============================================================================
# PHASE 7: RECEIVE -- Apply improvements
# ============================================================================

def apply_improvements(
    improvements: List[ProjectImprovement],
    scan: ProjectScan,
    auto: bool = False,
) -> List[ProjectImprovement]:
    """
    Apply approved improvements. Returns list of applied improvements.
    """
    _phase_log(6, "applying improvements...")

    if not improvements:
        return []

    applied = []

    if not auto:
        # Interactive approval
        print(f"\n  {BOLD}Apply improvements?{RESET}")
        print(f"  {DIM}a = all, n = none, or enter numbers (e.g., 1,3,5){RESET}")
        try:
            answer = input(f"  {LIME}>{RESET} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        if answer == "n" or answer == "":
            print(f"  {DIM}No improvements applied. Run again with --apply to apply later.{RESET}")
            return []

        if answer == "a":
            indices = list(range(len(improvements)))
        else:
            try:
                indices = [int(x.strip()) - 1 for x in answer.split(",")]
                indices = [i for i in indices if 0 <= i < len(improvements)]
            except ValueError:
                print(f"  {RED}Invalid input.{RESET}")
                return []
    else:
        # Auto mode: apply all with confidence >= 0.7 and auto_applicable
        indices = [
            i for i, imp in enumerate(improvements)
            if imp.confidence >= 0.7 and imp.auto_applicable
        ]

    for idx in indices:
        imp = improvements[idx]
        if imp.command:
            print(f"  {CYAN}Applying:{RESET} {imp.title}")
            ok, output = _run_cmd(
                ["sh", "-c", imp.command], scan.path, timeout=60
            )
            if ok:
                print(f"    {GREEN}OK{RESET}")
                applied.append(ProjectImprovement(
                    **{**asdict(imp), "applied": True,
                       "applied_at": datetime.now(timezone.utc).isoformat()}
                ))
            else:
                print(f"    {RED}FAILED{RESET}: {output[:100]}")
        else:
            print(f"  {YELLOW}Manual:{RESET} {imp.title}")
            print(f"    {DIM}{imp.description}{RESET}")
            applied.append(ProjectImprovement(
                **{**asdict(imp), "applied": True,
                   "applied_at": datetime.now(timezone.utc).isoformat()}
            ))

    return applied


# ============================================================================
# PHASE 8: IMPROVE -- Save state and learn from the evolution
# ============================================================================

def save_project_state(
    scan: ProjectScan,
    analysis: Dict,
    improvements: List[ProjectImprovement],
    applied: List[ProjectImprovement],
) -> None:
    """Save evolution state for this project."""
    _phase_log(7, "saving evolution state...")

    PROJECT_EVOLUTION_DIR.mkdir(parents=True, exist_ok=True)

    # Create a safe filename from project path
    safe_name = scan.name + "_" + str(hash(scan.path))[-8:]
    state_path = PROJECT_EVOLUTION_DIR / f"{safe_name}.json"

    # Load existing state or create new
    if state_path.exists():
        try:
            with open(state_path) as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    else:
        existing = {}

    state = {
        "project_path": scan.path,
        "project_name": scan.name,
        "scan_count": existing.get("scan_count", 0) + 1,
        "last_scan": scan.scanned_at,
        "health_score": analysis.get("health_score", 0.5),
        "improvements_suggested": existing.get("improvements_suggested", 0) + len(improvements),
        "improvements_applied": existing.get("improvements_applied", 0) + len(applied),
        "tech_stack": analysis.get("tech_stack", scan.ecosystems),
        "history": existing.get("history", []) + [{
            "scan_at": scan.scanned_at,
            "health_score": analysis.get("health_score", 0.5),
            "improvements_count": len(improvements),
            "applied_count": len(applied),
        }],
    }

    # Keep history to last 50 entries
    state["history"] = state["history"][-50:]

    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

    # Also persist improvements as knowledge atoms (learning from project evolution)
    _persist_learnings(scan, analysis, improvements)


def _persist_learnings(
    scan: ProjectScan,
    analysis: Dict,
    improvements: List[ProjectImprovement],
) -> None:
    """Persist project evolution learnings to WeEvolve knowledge base."""
    try:
        from weevolve.core import learn as we_learn

        # Create a learning from the project analysis
        learning_text = (
            f"Project evolution analysis of {scan.name} "
            f"({', '.join(scan.ecosystems)} project, {scan.file_count} files).\n"
            f"Health score: {analysis.get('health_score', 0.5):.0%}.\n"
            f"Found {len(improvements)} improvements.\n"
            f"Top improvement: {improvements[0].title if improvements else 'None'}.\n"
            f"Strengths: {', '.join(analysis.get('strengths', [])[:3])}."
        )

        we_learn(learning_text, source_type="text", verbose=False)
    except Exception:
        pass  # Silent -- learning persistence is bonus, not critical


# ============================================================================
# DAEMON MODE -- Recurring evolution checks
# ============================================================================

def run_project_daemon(
    project_path: Optional[str] = None,
    interval: int = 86400,  # Default: daily
    auto_apply: bool = False,
) -> None:
    """
    Run recurring project evolution checks.
    Default interval: 24 hours. Use --interval to customize.
    """
    base = Path(project_path) if project_path else Path.cwd()

    shutdown = False

    def _handle_sigint(_sig, _frame):
        nonlocal shutdown
        shutdown = True

    signal.signal(signal.SIGINT, _handle_sigint)

    print(f"\n{'=' * 60}")
    print(f"  (*) WeEvolve PROJECT DAEMON - Continuous Evolution")
    print(f"{'=' * 60}")
    print(f"  Project:   {base.name}")
    print(f"  Path:      {base}")
    print(f"  Interval:  {interval}s ({interval // 3600}h)")
    print(f"  Auto-apply: {auto_apply}")
    print(f"  Ctrl+C to stop")
    print(f"{'=' * 60}\n")

    cycle = 0
    while not shutdown:
        cycle += 1
        print(f"\n--- Evolution Cycle {cycle} ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ---\n")

        try:
            result = run_project_scan(
                project_path=str(base),
                do_apply=auto_apply,
                competitive=False,
                interactive=False,
            )

            if result:
                health = result.get("health_score", 0)
                improved = result.get("applied_count", 0)
                print(f"\n  Health: {health:.0%} | Applied: {improved} improvements")

        except Exception as exc:
            print(f"  {RED}[ERROR]{RESET} Evolution cycle failed: {exc}")

        if not shutdown:
            print(f"  Next evolution in {interval // 3600}h...")
            # Sleep in chunks so Ctrl+C responds quickly
            for _ in range(interval):
                if shutdown:
                    break
                time.sleep(1)

    print(f"\n  (*) Project daemon stopped. Evolution state saved.\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_project_scan(
    project_path: Optional[str] = None,
    do_apply: bool = False,
    competitive: bool = False,
    interactive: bool = True,
) -> Optional[Dict]:
    """
    Main project evolution pipeline.
    PERCEIVE -> CONNECT -> LEARN -> QUESTION -> EXPAND -> SHARE -> RECEIVE -> IMPROVE
    """
    print(f"\n{'=' * 60}")
    print(f"  (*) WeEvolve PROJECT EVOLUTION")
    print(f"  PERCEIVE -> CONNECT -> LEARN -> QUESTION -> IMPROVE")
    print(f"{'=' * 60}\n")

    # Phase 1: PERCEIVE
    scan = scan_project(project_path)
    print(f"  {DIM}Ecosystems: {', '.join(scan.ecosystems) or 'unknown'}{RESET}")
    print(f"  {DIM}Files: {scan.file_count} | Lines: {scan.total_lines:,} | Deps: {scan.dependency_count}{RESET}")
    print()

    # Phase 2: CONNECT (competitive analysis)
    competitors = []
    if competitive:
        competitors = find_competitors(scan)

    # Phase 3: LEARN (generate improvements)
    analysis = generate_improvements(scan, competitors)

    # Phase 4: QUESTION (validate)
    raw_improvements = analysis.get("improvements", [])
    improvements = validate_improvements(raw_improvements, scan)

    # Phase 5-6: EXPAND + SHARE (display)
    _phase_log(5, "sharing analysis")
    display_results(scan, analysis, improvements)

    # Phase 7: RECEIVE (apply if requested)
    applied = []
    if do_apply and improvements:
        if interactive:
            applied = apply_improvements(improvements, scan, auto=False)
        else:
            applied = apply_improvements(improvements, scan, auto=True)

    # Phase 8: IMPROVE (save state)
    save_project_state(scan, analysis, improvements, applied)

    print(f"\n  {LIME}(*){RESET} Project evolution complete.")
    if not do_apply and improvements:
        print(f"  {DIM}Run with --apply to apply improvements.{RESET}")
    print()

    return {
        "health_score": analysis.get("health_score", 0.5),
        "improvements_count": len(improvements),
        "applied_count": len(applied),
    }


def run_project(args: List[str]) -> None:
    """Parse arguments and run the appropriate project evolution mode."""
    project_path = None
    do_apply = "--apply" in args
    competitive = "--competitive" in args
    daemon_mode = "--daemon" in args

    if "--path" in args:
        idx = args.index("--path")
        if idx + 1 < len(args):
            project_path = args[idx + 1]

    if daemon_mode:
        interval = 86400  # 24h default
        if "--interval" in args:
            idx = args.index("--interval")
            if idx + 1 < len(args):
                try:
                    interval = int(args[idx + 1])
                except ValueError:
                    pass
        auto_apply = "--auto-apply" in args
        run_project_daemon(
            project_path=project_path,
            interval=interval,
            auto_apply=auto_apply,
        )
    else:
        run_project_scan(
            project_path=project_path,
            do_apply=do_apply,
            competitive=competitive,
        )
