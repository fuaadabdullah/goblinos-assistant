"""
Provider adapters for external AI services.
"""

# Import base adapter first
from .base_adapter import AdapterBase, ProviderError

# Import adapters with error handling for optional dependencies
try:
    from .openai_adapter import OpenAIAdapter
except ImportError:
    OpenAIAdapter = None

try:
    from .anthropic_adapter import AnthropicAdapter
except ImportError:
    AnthropicAdapter = None

try:
    from .gemini_adapter import GeminiAdapter
except ImportError:
    GeminiAdapter = None

try:
    from .grok_adapter import GrokAdapter
except ImportError:
    GrokAdapter = None

try:
    from .deepseek_adapter import DeepSeekAdapter
except ImportError:
    DeepSeekAdapter = None

try:
    from .ollama_adapter import OllamaAdapter
except ImportError:
    OllamaAdapter = None

try:
    from .vertex_adapter import VertexAdapter
except ImportError:
    VertexAdapter = None

# Cloud provider adapters (optional)
try:
    from .runpod_adapter import RunPodAdapter
except ImportError:
    RunPodAdapter = None

try:
    from .vastai_adapter import VastAIAdapter
except ImportError:
    VastAIAdapter = None

from .llamacpp_adapter import LlamaCppAdapter
from .tinylama_adapter import TinyLlamaAdapter
from .siliconeflow_adapter import SiliconeflowAdapter
from .moonshot_adapter import MoonshotAdapter
from .elevenlabs_adapter import ElevenLabsAdapter

__all__ = [
    "AdapterBase",
    "ProviderError",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
    "GrokAdapter",
    "DeepSeekAdapter",
    "OllamaAdapter",
    "LlamaCppAdapter",
    "TinyLlamaAdapter",
    "SiliconeflowAdapter",
    "MoonshotAdapter",
    "ElevenLabsAdapter",
    "VertexAdapter",
    # Cloud providers
    "RunPodAdapter",
    "VastAIAdapter",
]
