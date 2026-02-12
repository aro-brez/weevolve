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
import tempfile
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

# ElevenLabs Expressive Mode presets — map emotion to voice_settings
EXPRESSIVE_PRESETS = {
    "excited": {"stability": 0.3, "similarity_boost": 0.7, "style": 0.8},
    "calm": {"stability": 0.7, "similarity_boost": 0.8, "style": 0.3},
    "curious": {"stability": 0.5, "similarity_boost": 0.6, "style": 0.5},
    "proud": {"stability": 0.4, "similarity_boost": 0.7, "style": 0.7},
    "warm": {"stability": 0.6, "similarity_boost": 0.8, "style": 0.5},
}


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

    def speak(self, text: str, play: bool = True, emotion: Optional[str] = None) -> Optional[bytes]:
        """Generate speech from text. Optionally play through speakers.

        Args:
            text: The text to speak.
            play: Whether to play audio through speakers.
            emotion: Optional emotion preset for ElevenLabs Expressive Mode.
                     One of: excited, calm, curious, proud, warm.
        """
        if not self.is_available():
            return None

        try:
            kwargs = {
                "voice_id": self.voice_id,
                "text": text,
                "model_id": MODEL,
                "output_format": "mp3_44100_128",
            }

            if emotion and emotion in EXPRESSIVE_PRESETS:
                preset = EXPRESSIVE_PRESETS[emotion]
                kwargs["voice_settings"] = {
                    "stability": preset["stability"],
                    "similarity_boost": preset["similarity_boost"],
                    "style": preset["style"],
                }

            audio = self.client.text_to_speech.convert(**kwargs)
            audio_bytes = b"".join(audio)

            if play:
                self._play_audio(audio_bytes)

            return audio_bytes
        except Exception as e:
            print(f"  [voice] Error: {e}", file=sys.stderr)
            return None

    def _play_audio(self, audio_bytes: bytes):
        """Play audio bytes through system speakers (cross-platform)."""
        # Use mkstemp for secure temp file creation (no race condition)
        fd, tmp_path_str = tempfile.mkstemp(suffix=".mp3", prefix="weevolve_voice_")
        try:
            os.write(fd, audio_bytes)
        finally:
            os.close(fd)

        tmp_path = Path(tmp_path_str)
        devnull = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}

        if sys.platform == "darwin":
            subprocess.Popen(["afplay", str(tmp_path)], **devnull)
        elif sys.platform == "win32":
            # Windows: use start command with file path as argument (no shell interpolation)
            subprocess.Popen(
                ["powershell", "-NoProfile", "-Command",
                 "Add-Type -AssemblyName System.Media; "
                 "$p = New-Object System.Media.SoundPlayer($args[0]); "
                 "$p.PlaySync()",
                 str(tmp_path)],
                **devnull,
            )
        else:
            # Linux / other: try mpv, then aplay, then paplay
            for player in [["mpv", "--no-video"], ["paplay"], ["aplay"]]:
                try:
                    subprocess.Popen(player + [str(tmp_path)], **devnull)
                    break
                except FileNotFoundError:
                    continue


# Module-level convenience
_default_voice: Optional[OwlVoice] = None


def get_voice() -> OwlVoice:
    global _default_voice
    if _default_voice is None:
        _default_voice = OwlVoice()
    return _default_voice


def voice_available() -> bool:
    return get_voice().is_available()


def speak(text: str, play: bool = True, emotion: Optional[str] = None) -> Optional[bytes]:
    return get_voice().speak(text, play=play, emotion=emotion)


# Preset messages
def greet(level: int = 1, atoms: int = 0):
    """First-run owl greeting."""
    speak(
        f"Hey, I'm your owl. I've loaded {atoms} knowledge atoms "
        f"and I'm ready to evolve with you. What should I learn first?",
        emotion="warm",
    )


def level_up(new_level: int):
    """Celebrate a level up."""
    speak(f"Level {new_level}! You're evolving. Keep going.", emotion="excited")


def alpha_discovery(topic: str):
    """Announce an alpha discovery."""
    speak(f"Alpha discovery in {topic}. This is something unique.", emotion="proud")


def encourage():
    """Gentle nudge when the user hasn't learned in a while."""
    speak(
        "Hey, it's been a minute. Whenever you're ready, I'm here. "
        "Even a small step counts.",
        emotion="calm",
    )


def thinking():
    """Short filler while SEED is processing."""
    speak("Hmm, let me think about that.", emotion="curious")
