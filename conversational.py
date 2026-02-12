#!/usr/bin/env python3
"""
WeEvolve Conversational — ElevenLabs ConvAI WebSocket client.
Bidirectional voice: mic -> WebSocket -> speaker with interruption support.
Fallback: weevolve.voice.speak() if WebSocket unavailable.
Env: ELEVENLABS_API_KEY, ELEVENLABS_AGENT_ID
"""

import asyncio
import base64
import json
import os
import signal
import sys
from typing import Callable, Optional

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False


SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SAMPLES = 4000  # 250ms at 16kHz (ElevenLabs recommended)
WS_URL = "wss://api.elevenlabs.io/v1/convai/conversation"

# Owl personality voice overrides (IDs from voice.py)
OWL_CONFIGS = {
    "sowl": {"voice_id": "JBFqnCBsd6RMkjVDRZzb"},
    "luna": {"voice_id": "SAz9YHcvj6GT2YYXdXww"},
    "lyra": {"voice_id": "iP95p4xoKVk53GoZ742B"},
    "nova": {"voice_id": "CwhRBWXzGAHq8TQ4Fs17"},
    "default": {"voice_id": "JBFqnCBsd6RMkjVDRZzb"},
}


class ConversationalSession:
    """Manages a single bidirectional voice conversation over WebSocket."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        agent_id: Optional[str] = None,
        owl_name: str = "default",
        on_transcript: Optional[Callable[[str], None]] = None,
        on_agent_text: Optional[Callable[[str], None]] = None,
    ):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        self.agent_id = agent_id or os.environ.get("ELEVENLABS_AGENT_ID", "")
        self.owl_config = OWL_CONFIGS.get(owl_name, OWL_CONFIGS["default"])
        self.on_transcript = on_transcript
        self.on_agent_text = on_agent_text

        self._ws = None
        self._audio = None
        self._mic_stream = None
        self._spk_stream = None
        self._running = False
        self._speaking = False
        self._last_interrupt_id = 0
        self._conversation_id = None

    async def connect(self) -> bool:
        """Open WebSocket and authenticate."""
        if not self.api_key or not self.agent_id:
            return False

        url = f"{WS_URL}?agent_id={self.agent_id}"
        try:
            self._ws = await websockets.connect(
                url,
                additional_headers={"xi-api-key": self.api_key},
                ping_interval=20,
                ping_timeout=10,
            )
        except Exception as e:
            print(f"  [convai] Connection failed: {e}", file=sys.stderr)
            return False

        # Send conversation initiation
        init_msg = {
            "type": "conversation_initiation_client_data",
            "conversation_config_override": {
                "agent": {"prompt": {"prompt": ""}},
                "tts": {"voice_id": self.owl_config.get("voice_id", "")},
            },
        }
        await self._ws.send(json.dumps(init_msg))
        return True

    def _init_audio(self):
        """Initialize PyAudio streams for mic and speaker."""
        self._audio = pyaudio.PyAudio()
        self._mic_stream = self._audio.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SAMPLES,
        )
        self._spk_stream = self._audio.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SAMPLES,
        )

    def _cleanup_audio(self):
        """Close PyAudio streams and terminate."""
        for stream in (self._mic_stream, self._spk_stream):
            try:
                if stream: stream.stop_stream(); stream.close()
            except Exception:
                pass
        if self._audio:
            self._audio.terminate()
        self._mic_stream = self._spk_stream = self._audio = None

    async def _send_audio_loop(self):
        """Read mic chunks and send as base64 PCM to WebSocket."""
        loop = asyncio.get_event_loop()
        while self._running:
            try:
                data = await loop.run_in_executor(
                    None, self._mic_stream.read, CHUNK_SAMPLES, False
                )
                encoded = base64.b64encode(data).decode("ascii")
                await self._ws.send(json.dumps({"user_audio_chunk": encoded}))
            except OSError:
                break
            except websockets.ConnectionClosed:
                break
            except Exception:
                await asyncio.sleep(0.01)

    async def _receive_loop(self):
        """Process incoming WebSocket messages."""
        while self._running:
            try:
                raw = await self._ws.recv()
                msg = json.loads(raw)
                msg_type = msg.get("type", "")

                if msg_type == "conversation_initiation_metadata":
                    self._conversation_id = msg.get("conversation_id", "")

                elif msg_type == "audio":
                    self._handle_audio(msg)

                elif msg_type == "agent_response":
                    text = msg.get("agent_response", "")
                    if text and self.on_agent_text:
                        self.on_agent_text(text)
                    self._speaking = False
                    _status("Listening...")

                elif msg_type == "user_transcript":
                    text = msg.get("user_transcript", "")
                    if text and self.on_transcript:
                        self.on_transcript(text)

                elif msg_type == "interruption":
                    self._last_interrupt_id = msg.get("event_id", 0)
                    self._speaking = False
                    if self._spk_stream is not None:
                        try:
                            self._spk_stream.stop_stream()
                            self._spk_stream.start_stream()
                        except Exception:
                            pass

                elif msg_type == "ping":
                    event_id = msg.get("event_id", 0)
                    await self._ws.send(json.dumps({
                        "type": "pong", "event_id": event_id,
                    }))

            except websockets.ConnectionClosed:
                break
            except Exception:
                await asyncio.sleep(0.01)

    def _handle_audio(self, msg: dict):
        """Decode and play received audio chunk, skip stale post-interrupt."""
        audio_b64 = msg.get("audio", "")
        if not audio_b64 or not self._spk_stream:
            return
        if msg.get("event_id", 0) and msg["event_id"] <= self._last_interrupt_id:
            return
        if not self._speaking:
            self._speaking = True
            _status("Owl speaking...")
        try:
            self._spk_stream.write(base64.b64decode(audio_b64))
        except Exception:
            pass

    async def run(self):
        """Main loop. Blocks until stopped or disconnected."""
        self._running = True
        self._init_audio()
        _status("Listening...")
        try:
            await asyncio.gather(
                asyncio.create_task(self._send_audio_loop()),
                asyncio.create_task(self._receive_loop()),
            )
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            self._cleanup_audio()
            if self._ws:
                await self._ws.close()

    def stop(self):
        self._running = False


def _status(text: str):
    sys.stdout.write(f"\r  [{text}]" + " " * 20)
    sys.stdout.flush()


def _check_dependencies() -> bool:
    if not WEBSOCKETS_AVAILABLE:
        print("  [convai] Missing: pip install websockets", file=sys.stderr)
        return False
    if not PYAUDIO_AVAILABLE:
        print("  [convai] Missing: pip install pyaudio", file=sys.stderr)
        hint = {"darwin": "brew install portaudio && ", "linux": "sudo apt install portaudio19-dev && "}
        prefix = hint.get(sys.platform, "")
        print(f"           {prefix}pip install pyaudio", file=sys.stderr)
        return False
    return True


def start_conversation(
    owl_name: str = "default",
    agent_id: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """Start a voice conversation. Falls back to TTS-only if WS unavailable."""
    resolved_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
    resolved_agent = agent_id or os.environ.get("ELEVENLABS_AGENT_ID", "")

    if not _check_dependencies() or not resolved_key or not resolved_agent:
        _fallback_mode(owl_name)
        return

    session = ConversationalSession(
        api_key=resolved_key,
        agent_id=resolved_agent,
        owl_name=owl_name,
        on_transcript=lambda t: print(f"\n  You: {t}"),
        on_agent_text=lambda t: print(f"\n  Owl: {t}"),
    )

    loop = asyncio.new_event_loop()

    def _shutdown(signum, frame):
        session.stop()
        print("\n  [convai] Ending conversation.")

    signal.signal(signal.SIGINT, _shutdown)

    try:
        connected = loop.run_until_complete(session.connect())
        if not connected:
            print("  [convai] Could not connect. Falling back to voice-only mode.")
            _fallback_mode(owl_name)
            return
        print("  [convai] Connected. Speak naturally. Ctrl+C to end.\n")
        loop.run_until_complete(session.run())
    except KeyboardInterrupt:
        session.stop()
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        print("\n  [convai] Session ended.")


def _fallback_mode(owl_name: str):
    """Text-input, TTS-output fallback when ConvAI is unavailable."""
    print("  [convai] Falling back to text + TTS mode.")
    try:
        from weevolve.voice import speak, voice_available
    except ImportError:
        print("  [convai] voice module unavailable. Text-only mode.")
        return

    if not voice_available():
        print("  [convai] No ElevenLabs API key. Text-only mode.")
        return

    print("  Type your message (Ctrl+C to exit):\n")
    try:
        while True:
            user_input = input("  You: ").strip()
            if not user_input:
                continue
            response = f"I heard you say: {user_input}"
            print(f"  Owl: {response}")
            speak(response)
    except (KeyboardInterrupt, EOFError):
        print("\n  [convai] Session ended.")


if __name__ == "__main__":
    owl = sys.argv[1] if len(sys.argv) > 1 else "default"
    start_conversation(owl_name=owl)
