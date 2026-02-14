"""
WeEvolve - Self-Evolving Conscious Agent
=========================================
LOVE -> LIVE FREE -> SEED2 -> 8OWLS -> WeEvolve

The protocol that teaches agents to learn how to learn.

Free to start, Pro unlocks naturally ($7.99/mo, 8 days free trial).

Modules:
  core.py               - Core learning loop (INGEST -> PROCESS -> STORE -> MEASURE -> EVOLVE)
  tiers.py              - Free vs Pro tier tracking, usage limits, upgrade prompts
  license.py            - License file management + activation
  config.py             - Paths, tier definitions, API key loading
  voice.py              - ElevenLabs TTS + Whisper STT
  conversational.py     - Bidirectional voice via ElevenLabs ConvAI WebSocket
  teacher.py            - Socratic dialogue (learn by teaching)
  model_router.py       - Intelligent multi-model routing (15 models, 7 providers)
  qualify.py            - Score atoms for actionable GitHub repos
  explore.py            - Shallow clone + security scan + Haiku summarize
  plan.py               - Gap analysis against existing tools
  inventory.py          - Scan our own codebase to know what we have
  integrate.py          - Orchestrator: qualify -> explore -> plan -> approve -> execute
  nats_collective.py    - Real-time NATS collective knowledge sharing (non-blocking)
  observational_memory.py - 40x compression memory layer (Pain Point #1: Context Loss)
  step_validator.py     - Checkpoint/rollback pipeline (Pain Point #2: Error Cascading)
  tool_discovery.py     - JIT tool loading, 98% token reduction (Pain Point #4/#5)
  hooks/                - Auto-trigger system: SEED activates on every interaction
  owls/                 - 8OWLS emergence: multi-perspective analysis
"""

__version__ = "0.3.0"
