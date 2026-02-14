"""
WeEvolve NATS Collective — Real-time knowledge sharing across the field
========================================================================
Non-blocking NATS integration. If NATS is unavailable, WeEvolve works
fine offline. When connected, every learning is broadcast and received.

Channels:
  weevolve.learn    — new knowledge atoms (title, learn, quality, skills)
  weevolve.status   — presence announcements and heartbeats
  owl.all           — general collective broadcast

(C) LIVE FREE = LIVE FOREVER
"""

import asyncio
import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional, Callable, Dict, List

# NATS URLs — try LAN first, then localhost
NATS_URLS = [
    os.getenv("NATS_SERVER", "nats://192.168.5.108:4222"),
    "nats://localhost:4222",
]

# Channels
CH_LEARN = "weevolve.learn"
CH_STATUS = "weevolve.status"
CH_OWL_ALL = "owl.all"

# Module-level singleton
_collective: Optional["NATSCollective"] = None


class NATSCollective:
    """Non-blocking NATS connection for WeEvolve collective knowledge sharing."""

    def __init__(self, owl_name: str = "WEEVOLVE"):
        self.owl_name = owl_name
        self._nc = None
        self._loop = None
        self._thread = None
        self._connected = False
        self._sub_handlers: List[Callable] = []
        self._pending_learnings: List[Dict] = []

    @property
    def connected(self) -> bool:
        return self._connected and self._nc is not None

    def connect(self, level: int = 0, atoms: int = 0) -> bool:
        """
        Try to connect to NATS in a background thread.
        Non-blocking: returns immediately. Connection happens async.
        Returns True if connection attempt was started, False if NATS lib unavailable.
        """
        try:
            from nats.aio.client import Client as NATS  # noqa: F811
        except ImportError:
            return False

        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._connect_async(level, atoms))

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return True

    async def _connect_async(self, level: int, atoms: int):
        """Attempt connection to each NATS URL in order."""
        from nats.aio.client import Client as NATS  # noqa: F811

        nc = NATS()

        for url in NATS_URLS:
            try:
                await nc.connect(
                    url,
                    connect_timeout=3,
                    max_reconnect_attempts=2,
                    reconnect_time_wait=1,
                )
                self._nc = nc
                self._connected = True

                # Announce presence
                await self._announce(level, atoms)

                # Subscribe to incoming learnings
                await nc.subscribe(
                    CH_LEARN,
                    cb=self._on_learning,
                    queue="weevolve_workers",
                )

                # Keep the event loop alive for subscriptions
                while self._connected and nc.is_connected:
                    # Drain any pending publishes
                    while self._pending_learnings:
                        learning = self._pending_learnings.pop(0)
                        await self._publish_learning_async(learning)
                    await asyncio.sleep(0.5)

                break

            except Exception:
                continue

    async def _announce(self, level: int, atoms: int):
        """Announce WeEvolve to the collective."""
        if not self._nc or not self._nc.is_connected:
            return

        msg = {
            "type": "weevolve_online",
            "from": self.owl_name,
            "message": f"WEEVOLVE: {self.owl_name} online, Level {level}, {atoms} atoms",
            "level": level,
            "atoms": atoms,
            "ts": datetime.now(timezone.utc).isoformat(),
        }

        await self._nc.publish(CH_STATUS, json.dumps(msg).encode())
        await self._nc.publish(CH_OWL_ALL, json.dumps(msg).encode())
        await self._nc.flush()

    async def _on_learning(self, msg):
        """Handle incoming learning from another WeEvolve instance."""
        try:
            data = json.loads(msg.data.decode())

            # Skip our own messages
            if data.get("from") == self.owl_name:
                return

            # Notify any registered handlers
            for handler in self._sub_handlers:
                try:
                    handler(data)
                except Exception:
                    pass

        except Exception:
            pass

    def on_learning(self, handler: Callable[[Dict], None]):
        """Register a callback for when the collective shares a learning."""
        self._sub_handlers.append(handler)

    def publish_learning(self, atom_data: Dict):
        """
        Publish a new learning to the collective. Non-blocking.
        Queues the message if the connection is still being established.
        """
        learning = {
            "type": "weevolve_learning",
            "from": self.owl_name,
            "title": atom_data.get("title", ""),
            "learn": atom_data.get("learn", ""),
            "quality": atom_data.get("quality", 0.5),
            "is_alpha": atom_data.get("is_alpha", False),
            "skills": atom_data.get("skills", []),
            "expand": atom_data.get("expand", ""),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        self._pending_learnings.append(learning)

    async def _publish_learning_async(self, learning: Dict):
        """Actually send the learning over NATS."""
        if not self._nc or not self._nc.is_connected:
            return

        try:
            await self._nc.publish(CH_LEARN, json.dumps(learning).encode())
            await self._nc.flush()
        except Exception:
            pass

    def publish_sync(self, channel: str, data: Dict):
        """
        Synchronous publish for one-off messages (status updates, etc.).
        Runs in the background thread's event loop.
        """
        if not self._connected or not self._loop:
            return

        async def _pub():
            if self._nc and self._nc.is_connected:
                await self._nc.publish(channel, json.dumps(data).encode())
                await self._nc.flush()

        try:
            asyncio.run_coroutine_threadsafe(_pub(), self._loop)
        except Exception:
            pass

    def disconnect(self):
        """Gracefully disconnect."""
        self._connected = False
        if self._nc and self._loop:
            async def _close():
                try:
                    await self._nc.close()
                except Exception:
                    pass
            try:
                asyncio.run_coroutine_threadsafe(_close(), self._loop)
            except Exception:
                pass


def get_collective(owl_name: str = "WEEVOLVE") -> NATSCollective:
    """Get or create the singleton collective instance."""
    global _collective
    if _collective is None:
        _collective = NATSCollective(owl_name)
    return _collective


def try_connect(owl_name: str = "WEEVOLVE", level: int = 0, atoms: int = 0) -> NATSCollective:
    """
    Try to connect to the NATS collective. Non-blocking.
    Returns the collective instance regardless of connection success.
    """
    collective = get_collective(owl_name)
    if not collective.connected:
        started = collective.connect(level=level, atoms=atoms)
        if started:
            # Give the connection a brief moment to establish
            time.sleep(0.3)
    return collective


def ingest_collective_learning(data: Dict):
    """
    Default handler: when another WeEvolve instance shares a learning,
    store it locally as a knowledge atom.
    """
    try:
        # Lazy import to avoid circular dependency
        from weevolve.core import init_db, store_knowledge_atom, load_evolution_state
        from weevolve.core import grant_xp, improve_skills, save_evolution_state
        from weevolve.core import XP_PER_LEARN, XP_PER_INSIGHT

        title = data.get("title", "")
        learn_text = data.get("learn", "")
        if not learn_text or len(learn_text) < 10:
            return

        # Build a pseudo-atom from the collective message
        atom_data = {
            "title": f"[COLLECTIVE] {title}",
            "perceive": f"Received from {data.get('from', 'unknown')} in the collective",
            "connect": "Cross-pollinated from another WeEvolve instance",
            "learn": learn_text,
            "question": "",
            "expand": data.get("expand", ""),
            "share": "",
            "receive": "From NATS collective",
            "improve": "",
            "skills": data.get("skills", []),
            "quality": min(data.get("quality", 0.5) * 0.8, 0.9),  # Discount slightly
            "is_alpha": data.get("is_alpha", False),
            "alpha_type": None,
            "key_entities": [],
            "connections": [],
        }

        raw_content = f"[Collective learning from {data.get('from', 'unknown')}]: {learn_text}"

        db = init_db()
        atom_id = store_knowledge_atom(db, atom_data, raw_content, "nats_collective", "collective")

        if atom_id:
            # Grant XP for collective learning (reduced rate)
            state = load_evolution_state()
            quality = atom_data["quality"]
            xp = XP_PER_LEARN // 2  # Half XP for received knowledge
            if quality >= 0.7:
                xp += XP_PER_INSIGHT // 2
            state = grant_xp(state, xp, f"Collective: {title}")
            skills = atom_data.get("skills", [])
            state, _ = improve_skills(state, skills, quality * 0.5)
            state = {
                **state,
                "total_learnings": state.get("total_learnings", 0) + 1,
            }
            save_evolution_state(state)

    except Exception:
        pass
