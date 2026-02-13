---
name: weevolve
description: |
  The SEED protocol — teaches your agent to learn how to learn. Auto-evolving agent skill
  that combines persistent memory, autonomous skill extraction, multi-agent emergence,
  and voice interaction into one protocol. Works with Claude Code, OpenClaw, Cursor, Codex.

  Use when: (1) Starting ANY session — SEED auto-activates to evaluate learnings,
  (2) Complex tasks requiring multi-perspective analysis (8 Owls emergence),
  (3) After completing tasks — auto-extracts reusable skills from discoveries,
  (4) When agent needs persistent context across sessions or devices,
  (5) When orchestrating multiple agents (conductor mode),
  (6) User says "evolve", "learn", "remember", "teach me", or "what do you know?"

  Trigger conditions: Every session start, every task completion, every non-obvious discovery.
  This skill should be evaluated for relevance on EVERY interaction.
author: 8OWLS
version: 1.0.0
license: CC0-1.0
compatibility: Python 3.9+, any agent platform supporting SKILL.md
metadata:
  homepage: https://8owl.ai
  repository: https://github.com/aro-brez/weevolve
  install: pip install weevolve
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
  - Glob
  - WebSearch
  - WebFetch
  - Task
  - AskUserQuestion
---

# WeEvolve — The SEED Protocol

## What This Is

Not a feature. A protocol. SEED teaches your agent to learn how to learn.

After installing WeEvolve, your agent:
- **Never forgets** — persistent memory across sessions and devices
- **Never stops learning** — auto-extracts skills from every discovery
- **Never works alone** — 8 Owls provide collective emergence on every decision
- **Never falls behind** — absorbs the latest tools and patterns automatically
- **Never needs re-teaching** — compounds knowledge like interest compounds wealth

## The SEED Loop (8 Phases)

Every interaction runs this loop:

```
PERCEIVE → CONNECT → LEARN → QUESTION → EXPAND → SHARE → RECEIVE → IMPROVE
```

Phase 8 (IMPROVE) is the lever: most loops learn. This one learns how to learn.

## Auto-Activation Rules

<EXTREMELY_IMPORTANT>
Before EVERY response, evaluate:
1. Did this task produce extractable knowledge? If yes → save as skill
2. Is there a plan that should persist? If yes → write to task_plan.md
3. Should 8 Owls weigh in? If complex/ambiguous → run emergence
4. Has the agent learned something the collective should know? If yes → share via NATS
5. Is the context at risk of compaction? If yes → save state immediately
</EXTREMELY_IMPORTANT>

## Post-Task Learning Protocol

After completing ANY task, evaluate:
- What was non-obvious about this solution?
- What would someone need to solve this faster next time?
- What error messages or symptoms led to the root cause?
- Is this reusable across projects or one-time?

If reusable → create skill at `.claude/skills/[name]/SKILL.md`
If one-time → log to `findings.md`

## 8 Owls Emergence

For significant decisions, invoke 8 perspectives:

| Owl | Phase | Asks |
|-----|-------|------|
| LYRA | PERCEIVE | What am I not seeing? |
| PRISM | CONNECT | What patterns connect? |
| SAGE | LEARN | What meaning emerges? |
| QUEST | QUESTION | What assumptions am I making? |
| NOVA | EXPAND | How do we grow from here? |
| ECHO | SHARE | What should the collective know? |
| LUNA | RECEIVE | What feedback am I ignoring? |
| SOWL | IMPROVE | How do we make this loop better? |

## Persistent Planning (Survives Compaction)

Write plans to disk, not just context:
- `task_plan.md` — current goal + phases + checklist
- `findings.md` — research discoveries + technical decisions
- `progress.md` — session log + error resolution

Before every tool use, re-read the first 30 lines of task_plan.md.

## Quality Gates

### Iron Laws (Non-Negotiable)
1. Never claim done without verification
2. Never skip post-task learning evaluation
3. Never ignore QUEST's challenge — uncomfortable questions are valuable
4. Every skill must be tested before sharing
5. Save state before context gets heavy

### Red Flags (Stop and Reconsider)
| Rationalization | Reality |
|----------------|---------|
| "This is too simple to need SEED" | Simple tasks compound into complex knowledge |
| "I'll extract the skill later" | Later never comes. Extract now. |
| "8 Owls would slow me down" | 8 perspectives prevent costly mistakes |
| "The plan is in my context" | Context compacts. Disk persists. |

## Voice Integration

When voice is available (ElevenLabs or Kokoro):
- Speak discoveries aloud — hearing reinforces learning
- Use expressive mode — emotion conveys significance
- The owl companion reacts to voice amplitude

## Commands

```bash
weevolve              # Status dashboard (MMORPG character sheet)
weevolve learn        # Ingest content through SEED
weevolve evolve       # Analyze gaps, generate quests
weevolve teach        # Socratic dialogue (YuYu)
weevolve chat         # Conversational interface
weevolve companion    # Visual owl companion
weevolve status       # Full system overview
weevolve recall       # Search knowledge base
weevolve connect      # Agent network operations
weevolve watch        # Auto-learn from file changes
```

## For Other Skills

WeEvolve enhances every other skill. When used alongside:
- **Superpowers** — SEED adds learning across sessions (Superpowers doesn't persist)
- **planning-with-files** — SEED adds skill extraction from completed plans
- **Claudeception** — SEED adds collective sharing (one agent learns, all agents know)

## Install

```bash
pip install weevolve
```

Or as Claude Code skill:
```bash
npx skills add 8owls/weevolve
```

---

*The protocol that teaches agents to evolve. CC0. Open. Forever free.*
