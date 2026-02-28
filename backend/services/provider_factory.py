"""
Provider factory service for creating LLM adapters.
Handles adapter creation and provider configuration resolution.
"""

from typing import Dict, Any, Tuple, Optional
import os
import logging

try:
    from ..errors import raise_internal_error
except ImportError:
    from errors import raise_internal_error

logger = logging.getLogger(__name__)


def _coalesce(*values: Optional[str]) -> Optional[str]:
    """Return the first non-empty value."""
    for value in values:
        if value:
            return value
    return None


def _normalize_base_url(base_url: Optional[str]) -> Optional[str]:
    """Normalize base URLs to avoid duplicate /v1 segments."""
    if not base_url:
        return None
    normalized = base_url.strip().rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized


def _resolve_provider_kind(provider_name: Optional[str]) -> Optional[str]:
    """Map provider names to local provider kinds."""
    name = (provider_name or "").lower()
    if "ollama" in name:
        return "ollama"
    if "llamacpp" in name or "llama.cpp" in name:
        return "llamacpp"
    return None


def _resolve_local_llm_config(
    provider_info: Dict[str, Any], provider_kind: str
) -> Tuple[str, str]:
    """Resolve local LLM configuration based on provider type and environment."""
    provider_name = (provider_info.get("name") or "").lower()
    disable_local_ml = os.getenv("DISABLE_LOCAL_ML", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    use_local_llm = (
        not disable_local_ml
        and os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
    )
    gcp_hint = "gcp" in provider_name
    gcp_ollama_url = _coalesce(
        os.getenv("OLLAMA_GCP_BASE_URL"),
        os.getenv("OLLAMA_GCP_URL"),
    )
    gcp_llamacpp_url = _coalesce(
        os.getenv("LLAMACPP_GCP_BASE_URL"),
        os.getenv("LLAMACPP_GCP_URL"),
    )
    gcp_api_key = _coalesce(
        os.getenv("GCP_LLM_API_KEY"),
        os.getenv("LOCAL_LLM_API_KEY"),
    )

    if provider_kind == "ollama":
        base_url = _coalesce(
            gcp_ollama_url if gcp_hint else None,
            provider_info.get("base_url"),
            os.getenv("LOCAL_LLM_PROXY_URL") if use_local_llm else None,
            os.getenv("KAMATERA_LLM_URL") if not use_local_llm else None,
            os.getenv("OLLAMA_BASE_URL") if not disable_local_ml else None,
            "http://localhost:11434" if not disable_local_ml else None,
        )
        api_key = _coalesce(
            _coalesce(os.getenv("OLLAMA_GCP_API_KEY"), gcp_api_key)
            if gcp_hint
            else None,
            os.getenv("LOCAL_LLM_API_KEY") if use_local_llm else None,
            os.getenv("KAMATERA_LLM_API_KEY") if not use_local_llm else None,
            os.getenv("OLLAMA_API_KEY"),
            "",
        )
    else:
        base_url = _coalesce(
            gcp_llamacpp_url if gcp_hint else None,
            provider_info.get("base_url"),
            os.getenv("LOCAL_LLM_PROXY_URL") if not disable_local_ml else None,
            os.getenv("LLAMACPP_BASE_URL") if not disable_local_ml else None,
            "http://localhost:8080" if not disable_local_ml else None,
        )
        api_key = _coalesce(
            _coalesce(os.getenv("LLAMACPP_GCP_API_KEY"), gcp_api_key)
            if gcp_hint
            else None,
            os.getenv("LOCAL_LLM_API_KEY"),
            os.getenv("LLAMACPP_API_KEY"),
            "",
        )

    return api_key or "", _normalize_base_url(base_url) or ""


def create_local_adapter(
    provider_info: Dict[str, Any]
) -> Tuple[str, str, Any]:
    """Create adapter for local providers (Ollama, Llama.cpp)."""
    provider_kind = _resolve_provider_kind(provider_info.get("name"))
    if provider_kind not in {"ollama", "llamacpp"}:
        raise ValueError("Unsupported local provider")

    provider_metrics_name = (
        provider_info.get("name") or provider_kind or "unknown"
    ).lower()

    api_key, base_url = _resolve_local_llm_config(provider_info, provider_kind)

    # Import adapters dynamically to avoid circular imports
    try:
        from ..providers import OllamaAdapter, LlamaCppAdapter
    except ImportError:
        from providers import OllamaAdapter, LlamaCppAdapter

    adapter = (
        OllamaAdapter(api_key, base_url)
        if provider_kind == "ollama"
        else LlamaCppAdapter(api_key, base_url)
    )

    return provider_kind, provider_metrics_name, adapter


def create_cloud_adapter(provider_info: Dict[str, Any]):
    """Create adapter for cloud providers (OpenAI, Anthropic, etc.)."""
    provider_name = str(provider_info["name"]).lower().replace("-", "_")
    # Import adapters dynamically to avoid circular imports
    try:
        from ..providers import (
            OpenAIAdapter,
            AnthropicAdapter,
            GrokAdapter,
            DeepSeekAdapter,
            GeminiAdapter,
            MoonshotAdapter,
            ElevenLabsAdapter,
            SiliconeflowAdapter,
        )
    except ImportError:
        from providers import (
            OpenAIAdapter,
            AnthropicAdapter,
            GrokAdapter,
            DeepSeekAdapter,
            GeminiAdapter,
            MoonshotAdapter,
            ElevenLabsAdapter,
            SiliconeflowAdapter,
        )

    provider_adapters = {
        "openai": (OpenAIAdapter, ["OPENAI_API_KEY"], None),
        "anthropic": (AnthropicAdapter, ["ANTHROPIC_API_KEY"], None),
        "grok": (GrokAdapter, ["GROK_API_KEY", "GROQ_API_KEY"], "https://api.x.ai/v1"),
        "groq": (OpenAIAdapter, ["GROQ_API_KEY"], "https://api.groq.com/openai/v1"),
        "deepseek": (DeepSeekAdapter, ["DEEPSEEK_API_KEY"], None),
        "aliyun": (
            OpenAIAdapter,
            ["ALIYUN_MODEL_SERVER_KEY", "ALIYUN_API_KEY", "ALIYUN_KEY"],
            (
                os.getenv("ALIYUN_MODEL_SERVER_URL")
                or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
        ),
        "alibaba": (
            OpenAIAdapter,
            ["ALIYUN_MODEL_SERVER_KEY", "ALIYUN_API_KEY", "ALIYUN_KEY"],
            (
                os.getenv("ALIYUN_MODEL_SERVER_URL")
                or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
        ),
        "alibaba_cloud": (
            OpenAIAdapter,
            ["ALIYUN_MODEL_SERVER_KEY", "ALIYUN_API_KEY", "ALIYUN_KEY"],
            (
                os.getenv("ALIYUN_MODEL_SERVER_URL")
                or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
        ),
        "azure_openai": (
            OpenAIAdapter,
            ["AZURE_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_KEY"],
            (os.getenv("AZURE_OPENAI_ENDPOINT") or "https://{resource}.openai.azure.com"),
        ),
        "azure": (
            OpenAIAdapter,
            ["AZURE_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_KEY"],
            (os.getenv("AZURE_OPENAI_ENDPOINT") or "https://{resource}.openai.azure.com"),
        ),
        "gemini": (GeminiAdapter, ["GEMINI_API_KEY", "GOOGLE_API_KEY"], None),
        "moonshot": (MoonshotAdapter, ["MOONSHOT_API_KEY"], None),
        "elevenlabs": (ElevenLabsAdapter, ["ELEVENLABS_API_KEY"], None),
        # OpenAI-compatible providers
        "openrouter": (
            OpenAIAdapter,
            ["OPENROUTER_API_KEY"],
            (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"),
        ),
        "siliconeflow": (
            SiliconeflowAdapter,
            ["SILICONEFLOW_API_KEY", "SILICONFLOW_API_KEY"],
            (os.getenv("SILICONEFLOW_BASE_URL") or "https://api.siliconflow.com/v1"),
        ),
    }

    if provider_name not in provider_adapters:
        raise_internal_error(f"Provider {provider_info['name']} not yet implemented")

    adapter_class, api_key_envs, default_base_url = provider_adapters[provider_name]
    api_key = None
    for env_name in api_key_envs:
        value = (os.getenv(env_name) or "").strip()
        if value:
            api_key = value
            break

    if not api_key:
        raise_internal_error(
            f"API key not configured for provider {provider_info['name']}"
        )

    base_url = provider_info.get("base_url") or default_base_url
    adapter = adapter_class(api_key, base_url)
    return provider_name, adapter


def create_adapter(provider_info: Dict[str, Any]):
    """Create the appropriate adapter based on provider type."""
    try:
        provider_kind, provider_metrics_name, adapter = create_local_adapter(
            provider_info
        )
        is_local = True
    except ValueError:
        provider_name, adapter = create_cloud_adapter(provider_info)
        provider_metrics_name = provider_name
        is_local = False

    return is_local, provider_metrics_name, adapter


class ProviderFactory:
    """
    Thin wrapper so the factory can be injected as a service.
    """

    def create_adapter(self, provider_info: Dict[str, Any]):
        return create_adapter(provider_info)
