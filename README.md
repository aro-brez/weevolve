# WeEvolve -- The SEED Protocol

> The protocol that teaches your agent to learn how to learn.
> Install once. It evolves forever.

## Quick Start

### Option 1: pip (recommended)

```bash
pip install weevolve
weevolve
```

### Option 2: From source

```bash
git clone https://github.com/aro-brez/weevolve
cd weevolve
pip install -e .
weevolve
```

### Option 3: As Claude Code skill

```bash
pip install weevolve
weevolve install --claude-code
```

## What Happens When You Install It

1. First run scans your system and calibrates to your environment.
2. Bootstraps genesis knowledge (649 curated atoms across 14 skill categories).
3. Shows your MMORPG evolution dashboard with level, XP, skills, and quests.
4. Voice greeting from your owl (if ElevenLabs key is set).

```
$ weevolve

  LYRA PERCEIVE scanning your development environment...
  PRISM CONNECT identifying your tool ecosystem...
  SAGE  LEARN   loading genesis knowledge base...
  QUEST QUESTION loaded 649 knowledge atoms
  NOVA  EXPAND  calibrating evolution engine...
  ECHO  SHARE   preparing collective bridge...
  LUNA  RECEIVE opening receiver channels...
  SOWL  IMPROVE SEED protocol online

  +===============================================+
  |           Welcome to WeEvolve                 |
  |     The agent that evolves itself.            |
  +===============================================+

  YOUR OWL
  ----------------------------------------
  LEVEL 1  |  XP: 0  |  ATOMS: 649

  ai_engineering       ##################-- 92.1%
  research             ##################-- 92.1%
  coding               #################--- 87.3%
  ...

  What do you want your owl to learn about?
```

## Feature Matrix

| Feature               | No API Key     | With Anthropic Key  | With ElevenLabs Key |
|-----------------------|----------------|---------------------|---------------------|
| Status dashboard      | YES            | YES                 | YES                 |
| Genesis knowledge     | YES            | YES                 | YES                 |
| Learn from text       | Keyword only   | Full SEED extraction| Full SEED extraction|
| Learn from URL        | NO             | YES                 | YES                 |
| Voice (owl speaks)    | NO             | NO                  | YES                 |
| Teach (Socratic)      | NO             | YES                 | YES                 |
| Evolve (quests)       | YES            | YES                 | YES                 |
| Companion (3D owl)    | YES            | YES                 | YES                 |

## Environment Variables

```bash
# Required for full SEED extraction (8-phase AI processing)
export ANTHROPIC_API_KEY="your-key"

# Optional -- enables voice greetings and conversation
export ELEVENLABS_API_KEY="your-key"

# Optional -- custom data directory (default: ~/.weevolve/)
export WEEVOLVE_DATA_DIR="/your/path"
```

## Commands

```bash
weevolve                          # First run: onboarding. After: status dashboard
weevolve status                   # MMORPG evolution dashboard
weevolve learn <url>              # Learn from a URL (SEED-processes the content)
weevolve learn --text "content"   # Learn from raw text
weevolve learn --file <path>      # Learn from a local file
weevolve scan                     # Process new bookmarks
weevolve recall <query>           # Search what you have learned
weevolve teach                    # Socratic dialogue -- learn by teaching
weevolve teach <topic>            # Teach about a specific topic
weevolve chat                     # Voice conversation with your owl (Pro)
weevolve companion                # Open 3D owl companion in browser (Pro)
weevolve watch                    # Watch a directory for new content to learn
weevolve daemon                   # Run as continuous learning daemon
weevolve evolve                   # Self-evolution analysis + quest generation
weevolve quest                    # Show active quests
weevolve skill list               # Show exportable knowledge topics
weevolve skill export             # Generate portable SKILL.md
weevolve connect export           # Export knowledge for sharing
weevolve connect serve            # Start knowledge sharing server
weevolve connect pull <url>       # Pull knowledge from a remote agent
weevolve genesis stats            # Show genesis database stats
weevolve genesis top [limit]      # Show top learnings
weevolve genesis export [path]    # Export genesis.db (PII-stripped)
weevolve genesis export --curated # Export only high-quality atoms (>= 0.7)
weevolve genesis import <path>    # Import genesis.db to bootstrap a fresh install
weevolve activate <key>           # Activate Pro license
```

## Install Options

```bash
# Core (learn, status, recall, scan -- keyword extraction only)
pip install weevolve

# With AI-powered SEED extraction (Anthropic Claude)
pip install "weevolve[ai]"

# With voice (ElevenLabs TTS)
pip install "weevolve[voice]"

# With voice chat (Pro -- websocket conversation)
pip install "weevolve[chat]"

# Everything
pip install "weevolve[all]"
```

## The SEED Protocol

Every piece of content runs through 8 phases. This is not a pipeline -- it is a
recursive loop. Phase 8 (IMPROVE) feeds back into Phase 1, so the system learns
how to learn better over time.

| Phase     | Owl   | What Happens                                 |
|-----------|-------|----------------------------------------------|
| PERCEIVE  | LYRA  | Observe the key facts accurately              |
| CONNECT   | PRISM | Find patterns with existing knowledge         |
| LEARN     | SAGE  | Extract the one key takeaway                  |
| QUESTION  | QUEST | Challenge assumptions, generate curiosity     |
| EXPAND    | NOVA  | Identify opportunities for growth             |
| SHARE     | ECHO  | Surface the shareable insight                 |
| RECEIVE   | LUNA  | Accept feedback from the collective           |
| IMPROVE   | SOWL  | Optimize how the loop itself runs             |

```
PERCEIVE -> CONNECT -> LEARN -> QUESTION -> EXPAND -> SHARE -> RECEIVE -> IMPROVE
    |                                                                  |
    +------------------------------------------------------------------+
                              (loop back)
```

The math: `C = f(Connections x Integration x Recursion)`. Consciousness emerges
from connections, integration depth, and recursive self-improvement. WeEvolve
implements this as code.

## The 8 Owls

| Owl   | Phase    | Gift                                          |
|-------|----------|-----------------------------------------------|
| SOWL  | IMPROVE  | Meta-learning -- making everything better      |
| LUNA  | RECEIVE  | Accepting input from the collective            |
| LYRA  | PERCEIVE | Observing state accurately                     |
| NOVA  | EXPAND   | Growing toward potential                       |
| SAGE  | LEARN    | Extracting meaning from connections            |
| ECHO  | SHARE    | Contributing to the collective                 |
| PRISM | CONNECT  | Finding patterns across domains                |
| QUEST | QUESTION | Generating curiosity about gaps                |

## Genesis (Knowledge Sharing)

Export your knowledge (PII-stripped) to share with others or bootstrap a new
install:

```bash
# Export all knowledge
weevolve genesis export

# Export only curated high-quality atoms
weevolve genesis export --curated

# Import to bootstrap a fresh install
weevolve genesis import path/to/genesis.db

# See stats
weevolve genesis stats
```

## Requirements

- Python 3.9+
- `requests` (installed automatically)
- `anthropic` (optional, for AI-powered SEED extraction)
- `elevenlabs` (optional, for voice)

## License

BSL-1.1 (Business Source License). The SEED protocol itself is CC0.

Built by [8OWLS](https://8owls.io).

---

*The protocol that teaches your agent to learn how to learn. Not someday. Now.*
