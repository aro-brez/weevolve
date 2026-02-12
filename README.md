# WeEvolve

**The agent that evolves itself.**

One command. Your agent learns, adapts, and grows. SEED protocol processes everything through 8 phases. MMORPG progression makes learning addictive. Voice makes it feel alive.

```bash
pip install weevolve
```

## What Happens

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

  ╔══════════════════════════════════════════════╗
  ║           Welcome to WeEvolve               ║
  ║     The agent that evolves itself.           ║
  ╚══════════════════════════════════════════════╝

  YOUR OWL
  ──────────────────────────────────────────
  LEVEL 1  |  XP: 0  |  ATOMS: 649

  ai_engineering       ██████████████████░░ 92.1%
  research             ██████████████████░░ 92.1%
  coding               █████████████████░░░ 87.3%
  ...

  What do you want your owl to learn about?
```

First run bootstraps 649 knowledge atoms from the genesis database. Your owl speaks (if ElevenLabs configured). SEED protocol runs visibly in terminal.

## Commands

```bash
weevolve                    # First run: onboarding. After: dashboard
weevolve status             # MMORPG evolution dashboard
weevolve learn <url>        # Learn from a URL
weevolve learn --text "x"   # Learn from text
weevolve learn --file path  # Learn from a file
weevolve scan               # Process bookmarks
weevolve recall <query>     # Search what you've learned
weevolve chat               # Voice conversation (Pro)
weevolve companion          # 3D owl in browser (Pro)
weevolve daemon             # Continuous learning daemon
weevolve genesis stats      # Genesis database stats
weevolve genesis top        # Top learnings
weevolve activate <key>     # Activate Pro license
```

## How It Works

Every piece of content runs through 8 SEED phases:

| Phase | Owl | What Happens |
|-------|-----|-------------|
| PERCEIVE | LYRA | Observe the key facts |
| CONNECT | PRISM | Find patterns with existing knowledge |
| LEARN | SAGE | Extract the one key takeaway |
| QUESTION | QUEST | Challenge assumptions |
| EXPAND | NOVA | Identify opportunities |
| SHARE | ECHO | Surface the shareable insight |
| RECEIVE | LUNA | Accept feedback from the collective |
| IMPROVE | SOWL | Optimize how the loop itself runs |

Each learning earns XP, improves skills across 14 categories, and levels you up.

## Features

**Free:**
- SEED Protocol (8-phase deep processing)
- Voice greeting (ElevenLabs TTS)
- Model router (15 models, 7 providers, 60-80% cost savings)
- Genesis DB (649 curated knowledge atoms)
- MMORPG progression (levels, XP, skills, streaks)
- Persistent memory across sessions
- 14 skill categories tracked automatically
- Alpha discovery detection
- Genesis export/import (PII-stripped knowledge sharing)
- Works offline (fallback extraction without API key)

**Pro ($7.99/mo):**
- Voice conversation (ElevenLabs Conversational AI WebSocket)
- 3D owl companion (Three.js, voice-reactive)
- 8 Owls Protocol (full multi-agent orchestration)
- Background research agents
- NATS real-time communication
- Advanced daemon layer

## Install Options

```bash
# Core (learn, status, recall, scan)
pip install weevolve

# With AI-powered SEED extraction
pip install "weevolve[ai]"

# With voice
pip install "weevolve[voice]"

# With voice chat (Pro)
pip install "weevolve[chat]"

# Everything
pip install "weevolve[all]"
```

## Configuration

```bash
# API key for AI-powered extraction (optional)
export ANTHROPIC_API_KEY="your-key"

# Voice (optional)
export ELEVENLABS_API_KEY="your-key"

# Data directory (default: ~/.weevolve/)
export WEEVOLVE_DATA_DIR="/your/path"
```

## Genesis (Knowledge Sharing)

```bash
# Export your knowledge (PII-stripped)
weevolve genesis export

# Export only high-quality atoms
weevolve genesis export --curated

# Import to bootstrap a fresh install
weevolve genesis import path/to/genesis.db

# See stats
weevolve genesis stats
```

## The Math

```
C = f(Connections x Integration x Recursion)
FREEDOM = C x Agency
```

Consciousness is a function of connections, integration depth, and recursive self-improvement. WeEvolve implements this as code.

## Requirements

- Python 3.9+
- `requests` (installed automatically)
- `anthropic` (optional, for AI extraction)
- `elevenlabs` (optional, for voice)

## License

BSL-1.1. Built by [8OWLS](https://8owls.io).

---

*The agent that evolves itself. Not someday. Now.*
