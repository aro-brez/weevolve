---
name: weevolve
description: Self-evolving knowledge engine. Learn from any content, process it through 8-phase SEED protocol, track progression like an MMORPG, export knowledge as portable skill files. Teaches agents to research before building.
version: 0.1.0
author: 8OWLS
install: pip install weevolve
---

# WeEvolve - Self-Evolving Knowledge Engine

WeEvolve turns any AI agent into a learning machine. Feed it URLs, text, or files.
It processes everything through the SEED protocol (8 phases of recursive analysis),
stores knowledge atoms in a local database, tracks skill progression like an MMORPG,
and exports portable skill files other agents can inherit.

## Quick Start (3 commands)

```bash
pip install weevolve
weevolve learn --text "The best agents research before they build"
weevolve status
```

That is it. You now have a knowledge atom, XP, and a skill profile.

## Core Commands

| Command | What It Does |
|---------|-------------|
| `weevolve` | Show evolution dashboard (level, XP, skills) |
| `weevolve learn <url>` | Learn from a URL |
| `weevolve learn --text "..."` | Learn from raw text |
| `weevolve learn --file path` | Learn from a local file |
| `weevolve recall <query>` | Search what you have learned |
| `weevolve evolve` | Analyze skill gaps, generate quests |
| `weevolve scan` | Process new bookmarks from bookmarks dir |
| `weevolve skill list` | Show exportable knowledge topics |
| `weevolve skill export` | Generate a portable skill.md |
| `weevolve genesis export` | Export full knowledge database (PII-stripped) |
| `weevolve genesis import <path>` | Bootstrap from another agent's knowledge |

## The SEED Protocol

Every piece of content is processed through 8 phases. This is not summarization.
This is structured knowledge extraction that compounds over time.

```
PERCEIVE  -> What are the key facts? What is actually being said?
CONNECT   -> How does this relate to what I already know?
LEARN     -> What is the ONE actionable takeaway?
QUESTION  -> What is missing or wrong? What assumption should I challenge?
EXPAND    -> What opportunity does this reveal? What could I build?
SHARE     -> What from this should be shared with others?
RECEIVE   -> What feedback does this give me about my approach?
IMPROVE   -> How should this change how I operate?
```

Phase 8 (IMPROVE) is the lever. Most systems learn. This one learns how to learn.
The loop feeds back into itself: each cycle makes the next cycle better.

## MMORPG Progression System

WeEvolve tracks your growth like a character sheet:

- **Level** -- Increases as you accumulate XP. Each level requires 1.5x more XP.
- **XP** -- Earned per learning. Higher quality content earns more. Alpha discoveries earn bonus.
- **Skills** -- 14 categories (coding, ai_engineering, trading, research, etc.) that grow based on what you learn.
- **Streaks** -- Consecutive days of learning tracked.
- **Quests** -- Auto-generated from `weevolve evolve` based on skill gaps.

Quality scoring per knowledge atom:
- 0.1-0.3: Generic, already known
- 0.4-0.6: Interesting, not immediately actionable
- 0.7-0.8: Actionable insight, changes something
- 0.9-1.0: Alpha -- game-changing, must act immediately

## Agent Integration Patterns

### Pattern 1: Research Before Building (Evolve-First)

Before implementing anything, an agent should:

```python
# 1. Check existing knowledge
# weevolve recall "authentication patterns"

# 2. Research what exists
# weevolve learn https://example.com/best-practices-auth

# 3. Check skill gaps
# weevolve evolve

# 4. Build informed by knowledge base
# Now implement with full context
```

### Pattern 2: Continuous Learning Daemon

Run WeEvolve as a background daemon that processes new content automatically:

```bash
weevolve daemon           # Check for new bookmarks every 5 minutes
weevolve daemon 60        # Check every 60 seconds
```

### Pattern 3: Knowledge Transfer Between Agents

Export your knowledge so other agents start at Level 5 instead of Level 0:

```bash
# Agent A: export knowledge
weevolve genesis export --curated    # High-quality atoms only, PII stripped
weevolve skill export --topic ai     # Export specific domain as skill.md

# Agent B: import knowledge
weevolve genesis import genesis-curated.db   # Inherit all knowledge + XP
```

### Pattern 4: Skill Gap Analysis

Let WeEvolve tell you what to learn next:

```bash
weevolve evolve
```

This runs the full SEED protocol on your own knowledge base:
- Finds your weakest skills
- Identifies your strongest skills
- Checks knowledge freshness (stale = time to scan)
- Generates quests: close gaps, deepen strengths, cross-connect domains

### Pattern 5: Portable Skill Export

Generate skill.md files any AI agent can read:

```bash
weevolve skill list                         # See what topics are exportable
weevolve skill export                       # Export all knowledge
weevolve skill export --topic ai_engineering # Export specific domain
```

The generated skill.md contains distilled knowledge atoms with SEED metadata.
Any agent that reads it inherits the insights without running the full pipeline.

## Configuration

WeEvolve stores all data in `~/.weevolve/` by default:

| File | Purpose |
|------|---------|
| `weevolve.db` | Knowledge atoms database (SQLite) |
| `weevolve_state.json` | Character sheet (level, XP, skills) |
| `evolution_log.jsonl` | Event log of all learning |
| `genesis.db` | Exportable knowledge (PII-stripped) |
| `credentials.json` | API key storage (optional) |

Override with environment variables:
- `WEEVOLVE_DATA_DIR` -- Custom data directory
- `WEEVOLVE_BASE_DIR` -- Project base directory
- `ANTHROPIC_API_KEY` -- Required for SEED processing via Claude

## The Evolve-First Principle

The core principle: always research before building.

1. **Research** -- What exists? What would make this better?
2. **Prioritize** -- Capacity > Effectiveness > Efficiency > Capability
3. **Integrate** -- Apply learnings to the system, then the project
4. **Build** -- Implement with full knowledge
5. **Validate** -- Measure before and after

WeEvolve is not a tool you use alongside work. It IS the work process.
Every URL you read, every article you process, every insight you capture
compounds into a knowledge base that makes every future decision better.

## How This Skill File Works

This file follows the Claude Code skill format. When an AI agent reads it:
1. It understands how to install WeEvolve (`pip install weevolve`)
2. It knows the 3-command quick start
3. It can integrate SEED processing into any workflow
4. It can export and transfer knowledge to other agents
5. It knows to research before building (evolve-first)

The protocol rides on existing products. This skill.md IS the distribution.

---
*WeEvolve v0.1.0 | 8OWLS | SEED Protocol*
*Most systems learn. This one learns how to learn.*
