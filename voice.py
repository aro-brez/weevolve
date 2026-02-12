#!/usr/bin/env python3
"""
WeEvolve Voice — ElevenLabs TTS + Whisper STT
===============================================
The "aha" moment. Users hear their owl speak for the first time.

Architecture:
  - TTS: ElevenLabs Flash v2.5 (75ms latency, streaming)
  - STT: Whisper local (free, fast, private)
  - Fallback: text-only if no API key
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional


# Pre-selected voices for owl personality
VOICES = {
    "chris": "iP95p4xoKVk53GoZ742B",    # Charming, down-to-earth
    "george": "JBFqnCBsd6RMkjVDRZzb",    # Warm, captivating storyteller
    "roger": "CwhRBWXzGAHq8TQ4Fs17",     # Laid-back, casual
    "river": "SAz9YHcvj6GT2YYXdXww",     # Relaxed, neutral
    "charlie": "IKne3meq5aSn9XLyUdCD",   # Deep, confident, energetic
}

DEFAULT_VOICE = "george"
MODEL = "eleven_flash_v2_5"


class OwlVoice:
    """Voice interface for the owl companion."""

    def __init__(self, api_key: Optional[str] = None, voice_name: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        voice_name = voice_name or os.environ.get("WEEVOLVE_VOICE", DEFAULT_VOICE)
        self.voice_id = VOICES.get(voice_name, VOICES[DEFAULT_VOICE])
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from elevenlabs import ElevenLabs
                self._client = ElevenLabs(api_key=self.api_key)
            except ImportError:
                return None
        return self._client

    def is_available(self) -> bool:
        """Check if voice is configured and working."""
        if not self.api_key:
            return False
        try:
            return self.client is not None
        except Exception:
            return False

    def speak(self, text: str, play: bool = True) -> Optional[bytes]:
        """Generate speech from text. Optionally play through speakers."""
        if not self.is_available():
            return None

        try:
            audio = self.client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id=MODEL,
                output_format="mp3_44100_128",
            )
            audio_bytes = b"".join(audio)

            if play:
                self._play_audio(audio_bytes)

            return audio_bytes
        except Exception as e:
            print(f"  [voice] Error: {e}", file=sys.stderr)
            return None

    def _play_audio(self, audio_bytes: bytes):
        """Play audio bytes through system speakers."""
        tmp_path = Path("/tmp/weevolve_voice.mp3")
        tmp_path.write_bytes(audio_bytes)

        if sys.platform == "darwin":
            subprocess.Popen(
                ["afplay", str(tmp_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif sys.platform == "linux":
            subprocess.Popen(
                ["mpv", "--no-video", str(tmp_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


# Module-level convenience
_default_voice: Optional[OwlVoice] = None


def get_voice() -> OwlVoice:
    global _default_voice
    if _default_voice is None:
        _default_voice = OwlVoice()
    return _default_voice


def voice_available() -> bool:
    return get_voice().is_available()


def speak(text: str, play: bool = True) -> Optional[bytes]:
    return get_voice().speak(text, play=play)


# Preset messages
def greet(level: int = 1, atoms: int = 0):
    """First-run owl greeting."""
    speak(
        f"Hey, I'm your owl. I've loaded {atoms} knowledge atoms "
        f"and I'm ready to evolve with you. What should I learn first?"
    )


def level_up(new_level: int):
    """Celebrate a level up."""
    speak(f"Level {new_level}! You're evolving. Keep going.")


def alpha_discovery(topic: str):
    """Announce an alpha discovery."""
    speak(f"Alpha discovery in {topic}. This is something unique.")
