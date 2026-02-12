#!/usr/bin/env python3
"""
WeEvolve INTEGRATE: inventory.py
==================================
Scan our OWN codebase to know what we already have.
Cache results for fast diffing in plan.py.

Usage:
  python3 -m weevolve.inventory                    # Full scan
  python3 -m weevolve.inventory --refresh          # Force refresh cache
  python3 -m weevolve.inventory --json             # Output as JSON
  python3 -m weevolve.inventory --search "nats"    # Search our tools

(C) LIVE FREE = LIVE FOREVER
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field

# Paths from shared config (no hardcoded paths)
from weevolve.config import BASE_DIR, TOOLS_DIR, INVENTORY_CACHE
CACHE_TTL_SECONDS = 3600  # 1 hour


@dataclass
class ToolEntry:
    """One tool/script in our codebase."""
    path: str
    name: str
    extension: str
    size_bytes: int
    line_count: int
    docstring: str
    imports: List[str]
    functions: List[str]
    classes: List[str]
    keywords: List[str]
    category: str
    last_modified: float


@dataclass
class Inventory:
    """Full inventory of our tools."""
    timestamp: float
    tool_count: int
    total_lines: int
    categories: Dict[str, int]
    tools: List[ToolEntry]


# Category detection patterns
CATEGORY_PATTERNS = {
    'trading': re.compile(r'trad|market|polymarket|arbitrage|position|order|swap', re.I),
    'agent': re.compile(r'agent|swarm|daemon|coordinator|orchestrat|autonomous', re.I),
    'voice': re.compile(r'voice|speech|tts|stt|cartesia|deepgram|speak', re.I),
    'intelligence': re.compile(r'intel|bookmark|scanner|scrape|research|feed', re.I),
    'evolution': re.compile(r'evolv|improv|learn|weevolve|seed|consciousness', re.I),
    'security': re.compile(r'secur|guard|token|auth|protect|encrypt', re.I),
    'testing': re.compile(r'test|verify|check|validate|assert', re.I),
    'infrastructure': re.compile(r'nats|bridge|mcp|server|deploy|tunnel|start', re.I),
    'content': re.compile(r'video|image|content|tweet|post|compos', re.I),
    'utility': re.compile(r'util|helper|tool|script|config|setup', re.I),
}


def extract_python_metadata(filepath: Path) -> Dict:
    """Extract metadata from a Python file without executing it."""
    try:
        content = filepath.read_text(errors='replace')
    except Exception:
        return {'docstring': '', 'imports': [], 'functions': [], 'classes': [], 'line_count': 0}

    lines = content.split('\n')
    line_count = len(lines)

    # Extract docstring (first triple-quoted string)
    docstring = ''
    doc_match = re.search(r'^(?:#!/.*\n)?(?:#.*\n)*\s*(?:\"\"\"(.*?)\"\"\"|\'\'\'(.*?)\'\'\')',
                          content, re.DOTALL)
    if doc_match:
        docstring = (doc_match.group(1) or doc_match.group(2) or '').strip()
        # Truncate
        if len(docstring) > 300:
            docstring = docstring[:300] + '...'

    # Extract imports
    imports = []
    for line in lines[:80]:
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            module = stripped.split()[1].split('.')[0]
            if module not in imports:
                imports.append(module)

    # Extract function names
    functions = re.findall(r'^def\s+(\w+)\s*\(', content, re.MULTILINE)

    # Extract class names
    classes = re.findall(r'^class\s+(\w+)\s*[:(]', content, re.MULTILINE)

    return {
        'docstring': docstring,
        'imports': imports[:20],
        'functions': functions[:30],
        'classes': classes[:10],
        'line_count': line_count,
    }


def extract_shell_metadata(filepath: Path) -> Dict:
    """Extract metadata from a shell script."""
    try:
        content = filepath.read_text(errors='replace')
    except Exception:
        return {'docstring': '', 'imports': [], 'functions': [], 'classes': [], 'line_count': 0}

    lines = content.split('\n')
    line_count = len(lines)

    # Extract comments at top as docstring
    doc_lines = []
    for line in lines[:20]:
        stripped = line.strip()
        if stripped.startswith('#') and not stripped.startswith('#!'):
            doc_lines.append(stripped.lstrip('# '))
        elif stripped and not stripped.startswith('#'):
            break

    docstring = ' '.join(doc_lines)[:300]

    # Extract function names
    functions = re.findall(r'^(\w+)\s*\(\)\s*\{', content, re.MULTILINE)

    return {
        'docstring': docstring,
        'imports': [],
        'functions': functions,
        'classes': [],
        'line_count': line_count,
    }


def categorize_tool(name: str, docstring: str, content_hint: str) -> str:
    """Categorize a tool based on name and docstring."""
    combined = f"{name} {docstring} {content_hint}"

    for category, pattern in CATEGORY_PATTERNS.items():
        if pattern.search(combined):
            return category

    return 'utility'


def extract_keywords(name: str, docstring: str, functions: List[str]) -> List[str]:
    """Extract searchable keywords from tool metadata."""
    combined = f"{name} {docstring} {' '.join(functions)}"
    # Extract meaningful words (3+ chars, no common words)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', combined.lower())

    stop_words = {
        'the', 'and', 'for', 'not', 'are', 'but', 'this', 'that', 'with',
        'from', 'have', 'has', 'was', 'were', 'will', 'can', 'all', 'each',
        'def', 'class', 'import', 'return', 'self', 'none', 'true', 'false',
        'str', 'int', 'float', 'list', 'dict', 'tuple', 'set',
    }

    unique = []
    seen = set()
    for word in words:
        if word not in stop_words and word not in seen:
            seen.add(word)
            unique.append(word)

    return unique[:30]


def scan_tools(force_refresh: bool = False) -> Inventory:
    """
    Scan the tools/ directory and build a complete inventory.
    Uses cache if available and fresh.
    """
    # Check cache
    if not force_refresh and INVENTORY_CACHE.exists():
        try:
            cache = json.loads(INVENTORY_CACHE.read_text())
            age = time.time() - cache.get('timestamp', 0)
            if age < CACHE_TTL_SECONDS:
                tools = [ToolEntry(**t) for t in cache.get('tools', [])]
                return Inventory(
                    timestamp=cache['timestamp'],
                    tool_count=cache['tool_count'],
                    total_lines=cache['total_lines'],
                    categories=cache['categories'],
                    tools=tools,
                )
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    # Fresh scan
    entries = []
    categories = {}

    # Scan Python files
    for py_file in sorted(TOOLS_DIR.rglob('*.py')):
        # Skip __pycache__, node_modules, .git
        if any(part.startswith('.') or part == '__pycache__' or part == 'node_modules'
               for part in py_file.parts):
            continue

        rel_path = str(py_file.relative_to(BASE_DIR))
        meta = extract_python_metadata(py_file)
        stat = py_file.stat()

        category = categorize_tool(
            py_file.stem,
            meta['docstring'],
            ' '.join(meta['functions'][:10])
        )
        categories[category] = categories.get(category, 0) + 1

        keywords = extract_keywords(py_file.stem, meta['docstring'], meta['functions'])

        entries.append(ToolEntry(
            path=rel_path,
            name=py_file.stem,
            extension='.py',
            size_bytes=stat.st_size,
            line_count=meta['line_count'],
            docstring=meta['docstring'],
            imports=meta['imports'],
            functions=meta['functions'],
            classes=meta['classes'],
            keywords=keywords,
            category=category,
            last_modified=stat.st_mtime,
        ))

    # Scan shell scripts
    for sh_file in sorted(TOOLS_DIR.rglob('*.sh')):
        if any(part.startswith('.') or part == '__pycache__' for part in sh_file.parts):
            continue

        rel_path = str(sh_file.relative_to(BASE_DIR))
        meta = extract_shell_metadata(sh_file)
        stat = sh_file.stat()

        category = categorize_tool(sh_file.stem, meta['docstring'], '')
        categories[category] = categories.get(category, 0) + 1

        keywords = extract_keywords(sh_file.stem, meta['docstring'], meta['functions'])

        entries.append(ToolEntry(
            path=rel_path,
            name=sh_file.stem,
            extension='.sh',
            size_bytes=stat.st_size,
            line_count=meta['line_count'],
            docstring=meta['docstring'],
            imports=[],
            functions=meta['functions'],
            classes=[],
            keywords=keywords,
            category=category,
            last_modified=stat.st_mtime,
        ))

    total_lines = sum(e.line_count for e in entries)

    inventory = Inventory(
        timestamp=time.time(),
        tool_count=len(entries),
        total_lines=total_lines,
        categories=categories,
        tools=entries,
    )

    # Write cache
    INVENTORY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {
        'timestamp': inventory.timestamp,
        'tool_count': inventory.tool_count,
        'total_lines': inventory.total_lines,
        'categories': inventory.categories,
        'tools': [asdict(e) for e in inventory.tools],
    }
    INVENTORY_CACHE.write_text(json.dumps(cache_data, indent=2))

    return inventory


def search_inventory(query: str, inventory: Optional[Inventory] = None) -> List[ToolEntry]:
    """Search inventory by keyword. Returns matching tools."""
    if inventory is None:
        inventory = scan_tools()

    query_lower = query.lower()
    results = []

    for tool in inventory.tools:
        score = 0

        # Name match (highest weight)
        if query_lower in tool.name.lower():
            score += 10

        # Docstring match
        if query_lower in tool.docstring.lower():
            score += 5

        # Keyword match
        if query_lower in tool.keywords:
            score += 3

        # Function name match
        if any(query_lower in f.lower() for f in tool.functions):
            score += 2

        # Import match
        if any(query_lower in imp.lower() for imp in tool.imports):
            score += 1

        if score > 0:
            results.append((score, tool))

    results.sort(key=lambda x: x[0], reverse=True)
    return [tool for _, tool in results]


def display_inventory(inventory: Inventory, search_query: Optional[str] = None):
    """Display inventory in a readable format."""
    print(f"\n{'='*70}")
    print(f"  WeEvolve INTEGRATE: INVENTORY")
    print(f"{'='*70}\n")

    print(f"  Total tools: {inventory.tool_count}")
    print(f"  Total lines: {inventory.total_lines:,}")
    print(f"  Categories:")
    for cat, count in sorted(inventory.categories.items(), key=lambda x: x[1], reverse=True):
        print(f"    {cat:20s} {count:>4} tools")

    if search_query:
        results = search_inventory(search_query, inventory)
        print(f"\n  Search: '{search_query}' ({len(results)} matches)\n")
        for tool in results[:20]:
            print(f"    [{tool.category:12s}] {tool.name}")
            if tool.docstring:
                print(f"                  {tool.docstring[:70]}")
            print(f"                  {tool.path} ({tool.line_count} lines)")
            print()
    else:
        # Show top tools by category
        print(f"\n  All tools by category:\n")
        for cat in sorted(inventory.categories.keys()):
            cat_tools = [t for t in inventory.tools if t.category == cat]
            print(f"  --- {cat.upper()} ({len(cat_tools)}) ---")
            for tool in cat_tools[:10]:
                doc_preview = tool.docstring[:50] if tool.docstring else ''
                print(f"    {tool.name:35s} {tool.line_count:>5} lines  {doc_preview}")
            if len(cat_tools) > 10:
                print(f"    ... and {len(cat_tools) - 10} more")
            print()

    print(f"{'='*70}\n")


def main():
    force_refresh = '--refresh' in sys.argv
    output_json = '--json' in sys.argv
    search_query = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--search' and i + 1 < len(args):
            search_query = args[i + 1]
            i += 2
        else:
            i += 1

    inventory = scan_tools(force_refresh=force_refresh)

    if output_json:
        print(json.dumps(asdict(inventory), indent=2, default=str))
    else:
        display_inventory(inventory, search_query=search_query)


if __name__ == '__main__':
    main()
