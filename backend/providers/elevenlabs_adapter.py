"""
ElevenLabs Text-to-Speech Provider Adapter for Goblin Assistant.

This adapter provides text-to-speech capabilities using ElevenLabs API.
Supports voice synthesis, voice cloning, and multi-language support.
"""

import os
from typing import Dict, Any, Optional, AsyncIterator
import aiohttp
import logging

logger = logging.getLogger(__name__)


class ElevenLabsAdapter:
    """
    ElevenLabs text-to-speech provider adapter.

    Capabilities:
    - High-quality voice synthesis
    - Multiple voice options
    - Multi-language support
    - Streaming audio generation
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize ElevenLabs adapter.

        Args:
            api_key: ElevenLabs API key (defaults to ELEVENLABS_API_KEY env var)
            base_url: Optional custom base URL (defaults to ElevenLabs API)
        """
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ElevenLabs API key is required")

        self.base_url = base_url or "https://api.elevenlabs.io/v1"
        self.headers = {"xi-api-key": self.api_key, "Content-Type": "application/json"}

        # Default voice ID (George - narrator voice)
        self.default_voice_id = "JBFqnCBsd6RMkjVDRZzb"

        # Default settings
        self.default_model_id = "eleven_multilingual_v2"
        self.default_output_format = "mp3_44100_128"

    async def generate(self, messages: list, **kwargs) -> Dict[str, Any]:
        """
        Generate speech from text using ElevenLabs API.

        Args:
            messages: List of message dicts with 'content' to synthesize
            **kwargs: Additional parameters (voice_id, model_id, output_format, etc.)

        Returns:
            Dict with audio data and metadata
        """
        # Extract text from messages
        text = self._extract_text_from_messages(messages)

        # Get parameters
        voice_id = kwargs.get("voice_id", self.default_voice_id)
        model_id = kwargs.get("model_id", self.default_model_id)
        output_format = kwargs.get("output_format", self.default_output_format)

        # Build request payload
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": kwargs.get("stability", 0.5),
                "similarity_boost": kwargs.get("similarity_boost", 0.75),
                "style": kwargs.get("style", 0.0),
                "use_speaker_boost": kwargs.get("use_speaker_boost", True),
            },
        }

        url = f"{self.base_url}/text-to-speech/{voice_id}"

        # Add output format as query parameter
        url = f"{url}?output_format={output_format}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, headers=self.headers, json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"ElevenLabs API error: {response.status} - {error_text}"
                    )

                audio_data = await response.read()

                return {
                    "audio": audio_data,
                    "format": output_format,
                    "voice_id": voice_id,
                    "model_id": model_id,
                    "text": text,
                    "provider": "elevenlabs",
                    "size_bytes": len(audio_data),
                }

    async def stream_generate(self, messages: list, **kwargs) -> AsyncIterator[bytes]:
        """
        Stream audio generation from ElevenLabs API.

        Args:
            messages: List of message dicts with 'content' to synthesize
            **kwargs: Additional parameters

        Yields:
            Audio data chunks as bytes
        """
        text = self._extract_text_from_messages(messages)
        voice_id = kwargs.get("voice_id", self.default_voice_id)
        model_id = kwargs.get("model_id", self.default_model_id)
        output_format = kwargs.get("output_format", self.default_output_format)

        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": kwargs.get("stability", 0.5),
                "similarity_boost": kwargs.get("similarity_boost", 0.75),
                "style": kwargs.get("style", 0.0),
                "use_speaker_boost": kwargs.get("use_speaker_boost", True),
            },
        }

        url = f"{self.base_url}/text-to-speech/{voice_id}/stream"
        url = f"{url}?output_format={output_format}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, headers=self.headers, json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"ElevenLabs API error: {response.status} - {error_text}"
                    )

                async for chunk in response.content.iter_chunked(1024):
                    yield chunk

    async def list_voices(self) -> Dict[str, Any]:
        """
        List available voices from ElevenLabs.

        Returns:
            Dict with voices list and metadata
        """
        url = f"{self.base_url}/voices"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"ElevenLabs API error: {response.status} - {error_text}"
                    )

                data = await response.json()
                return data

    async def get_voice_details(self, voice_id: str) -> Dict[str, Any]:
        """
        Get details for a specific voice.

        Args:
            voice_id: Voice ID to query

        Returns:
            Dict with voice details
        """
        url = f"{self.base_url}/voices/{voice_id}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"ElevenLabs API error: {response.status} - {error_text}"
                    )

                data = await response.json()
                return data

    async def health_check(self) -> bool:
        """
        Check if ElevenLabs API is accessible.

        Returns:
            True if API is healthy, False otherwise
        """
        try:
            voices = await self.list_voices()
            return len(voices.get("voices", [])) > 0
        except Exception:
            return False

    def _extract_text_from_messages(self, messages: list) -> str:
        """
        Extract text content from messages list.

        Args:
            messages: List of message dicts

        Returns:
            Combined text string
        """
        text_parts = []
        for msg in messages:
            if isinstance(msg, dict) and "content" in msg:
                text_parts.append(msg["content"])
            elif isinstance(msg, str):
                text_parts.append(msg)

        return " ".join(text_parts)

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get adapter capabilities.

        Returns:
            Dict describing supported features
        """
        return {
            "provider": "elevenlabs",
            "type": "text-to-speech",
            "streaming": True,
            "voice_cloning": True,
            "multi_language": True,
            "output_formats": [
                "mp3_44100_128",
                "mp3_44100_192",
                "pcm_16000",
                "pcm_22050",
                "pcm_24000",
                "pcm_44100",
                "ulaw_8000",
            ],
            "models": [
                "eleven_multilingual_v2",
                "eleven_monolingual_v1",
                "eleven_turbo_v2",
            ],
            "default_voice": self.default_voice_id,
        }
