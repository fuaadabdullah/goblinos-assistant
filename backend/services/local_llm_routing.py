"""
Local LLM Routing Strategy Configuration

This module defines intelligent routing rules for local Goblin Ollama Server models
based on intent, context length, latency requirements, and cost priorities.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


class Intent(str, Enum):
    """Request intent types for routing decisions."""

    SUMMARIZE = "summarize"
    EXPLAIN = "explain"
    CODE_GEN = "code-gen"
    CREATIVE = "creative"
    RETRIEVAL = "retrieval"
    RAG = "rag"
    CHAT = "chat"
    CLASSIFICATION = "classification"
    STATUS = "status"
    MICROOP = "microop"
    LEGAL = "legal"
    TRANSLATION = "translation"


class LatencyTarget(str, Enum):
    """Latency SLA targets."""

    ULTRA_LOW = "ultra_low"  # < 100ms
    LOW = "low"  # 100-200ms
    MEDIUM = "medium"  # 200-500ms
    HIGH = "high"  # 500ms+


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    model_id: str
    provider: str  # canonical provider id (e.g. "ollama_gcp", "llamacpp_gcp")
    context_window: int
    best_for: List[str]
    temperature: float
    top_p: float
    max_tokens: int
    stop_sequences: Optional[List[str]] = None


# Model routing configurations
MODEL_CONFIGS = {
    "mistral:7b": ModelConfig(
        model_id="mistral:7b",
        provider="ollama_gcp",
        context_window=8192,
        best_for=[
            "high_quality",
            "creative",
            "coding",
            "legal",
            "explain",
            "summarize",
        ],
        temperature=0.2,
        top_p=0.95,
        max_tokens=512,
        stop_sequences=["\n\n"],
    ),
    "qwen2.5:3b": ModelConfig(
        model_id="qwen2.5:3b",
        provider="ollama_gcp",
        context_window=32768,
        best_for=["long_context", "multilingual", "rag", "retrieval"],
        temperature=0.0,
        top_p=0.9,
        max_tokens=1024,
        stop_sequences=None,
    ),
    "phi3:3.8b": ModelConfig(
        model_id="phi3:3.8b",
        provider="ollama_gcp",
        context_window=4096,
        best_for=["low_latency", "chat", "conversational", "confidence_scoring"],
        temperature=0.15,
        top_p=0.9,
        max_tokens=128,
        stop_sequences=None,
    ),
    "gemma:2b": ModelConfig(
        model_id="gemma:2b",
        provider="ollama_gcp",
        context_window=8192,
        best_for=[
            "ultra_fast",
            "classification",
            "status",
            "microop",
            "safety_verification",
        ],
        temperature=0.0,
        top_p=0.9,
        max_tokens=40,
        stop_sequences=None,
    ),
    "goblin-simple-llama-1b": ModelConfig(
        model_id="goblin-simple-llama-1b",
        provider="ollama_gcp",
        context_window=2048,
        best_for=[
            "emergency_fallback",
            "rate_limited",
            "cheap_inference",
            "overload_protection",
        ],
        temperature=0.1,
        top_p=0.9,
        max_tokens=128,
        stop_sequences=None,
    ),
}


# System prompt templates by use case
SYSTEM_PROMPTS = {
    "default": (
        "You are a concise, accurate assistant. Use numbered steps for procedures. "
        "If unsure, say 'I don't know — check sources.' "
        "Do not invent facts; if information depends on external sources label it."
    ),
    "creative": (
        "You are a creative and imaginative assistant. Be expressive while remaining helpful. "
        "Do not invent facts; if information depends on external sources label it."
    ),
    "code": (
        "You are a precise coding assistant. Provide clean, working code with brief explanations. "
        "Use best practices and include error handling. "
        "Do not invent facts; if information depends on external sources label it."
    ),
    "rag": (
        "You are a retrieval assistant. Answer based strictly on provided context. "
        "If the answer is not in the context, say 'This information is not available in the provided context.' "
        "Do not invent facts; cite sources when available."
    ),
    "classification": (
        "You are a classification assistant. Provide only the requested classification without explanation. "
        "Be precise and consistent."
    ),
}


def estimate_token_count(text: str) -> int:
    """
    Rough token count estimation (1 token ≈ 4 characters for English).
    For production, use tiktoken or similar.
    """
    return len(text) // 4


def get_context_length(messages: List[Dict[str, str]]) -> int:
    """Calculate total context length from messages."""
    total_text = " ".join(msg.get("content", "") for msg in messages)
    return estimate_token_count(total_text)


def detect_intent(messages: List[Dict[str, str]]) -> Intent:
    """
    Detect intent from messages.
    Simple keyword-based detection - can be enhanced with a classifier.
    """
    last_message = messages[-1].get("content", "").lower() if messages else ""

    # Check each intent type
    if check_summarize_keywords(last_message):
        return Intent.SUMMARIZE
    elif check_explain_keywords(last_message):
        return Intent.EXPLAIN
    elif check_code_gen_keywords(last_message):
        return Intent.CODE_GEN
    elif check_creative_keywords(last_message):
        return Intent.CREATIVE
    elif check_translation_keywords(last_message):
        return Intent.TRANSLATION
    elif check_classification_keywords(last_message):
        return Intent.CLASSIFICATION
    elif check_status_keywords(last_message):
        return Intent.STATUS
    else:
        return Intent.CHAT


def check_summarize_keywords(text: str) -> bool:
    """Check if text contains summarize-related keywords."""
    return any(kw in text for kw in ["summarize", "summary", "tldr", "sum up"])


def check_explain_keywords(text: str) -> bool:
    """Check if text contains explain-related keywords."""
    return any(kw in text for kw in ["explain", "what is", "what does", "how does"])


def check_code_gen_keywords(text: str) -> bool:
    """Check if text contains code generation keywords."""
    return any(
        kw in text for kw in ["code", "function", "class", "implement", "script"]
    )


def check_creative_keywords(text: str) -> bool:
    """Check if text contains creative writing keywords."""
    return any(kw in text for kw in ["story", "poem", "creative", "imagine"])


def check_translation_keywords(text: str) -> bool:
    """Check if text contains translation keywords."""
    return any(kw in text for kw in ["translate", "translation", "say in"])


def check_classification_keywords(text: str) -> bool:
    """Check if text contains classification keywords."""
    return any(kw in text for kw in ["classify", "category", "label"])


def check_status_keywords(text: str) -> bool:
    """Check if text contains status/health check keywords."""
    return any(kw in text for kw in ["status", "health", "check"])


def detect_language(text: str) -> str:
    """
    Detect if text is primarily non-English.
    Simple heuristic - can be enhanced with langdetect or similar.
    """
    # Check for common non-ASCII characters
    non_ascii = sum(1 for char in text if ord(char) > 127)
    if non_ascii > len(text) * 0.3:  # More than 30% non-ASCII
        return "non-en"
    return "en"


def select_model(
    messages: List[Dict[str, str]],
    intent: Optional[Intent] = None,
    latency_target: LatencyTarget = LatencyTarget.MEDIUM,
    context_provided: Optional[str] = None,
    cost_priority: bool = False,
    force_cheap_fallback: bool = False,
) -> tuple[str, Dict[str, Any]]:
    """
    Select the best model based on routing rules.

    Args:
        messages: List of message dictionaries
        intent: Detected intent (auto-detected if None)
        latency_target: Latency requirements
        context_provided: Additional context string
        cost_priority: Prefer cheaper models
        force_cheap_fallback: Force use of cheap emergency model

    Returns:
        tuple: (model_id, parameters)
    """
    # Emergency fallback - use cheap model regardless of other rules
    if force_cheap_fallback:
        return get_cheap_fallback_model_config()

    # Auto-detect intent if not provided
    if intent is None:
        intent = detect_intent(messages)

    # Calculate context length
    context_length = get_context_length(messages)
    if context_provided:
        context_length += estimate_token_count(context_provided)

    # Detect language
    last_message = messages[-1].get("content", "") if messages else ""
    language = detect_language(last_message)

    # Apply routing rules in priority order
    if should_use_gemma_model(intent, latency_target, cost_priority, context_length):
        return get_gemma_model_config()
    elif should_use_qwen_model(context_length, language, intent):
        return get_qwen_model_config(intent)
    elif should_use_phi3_model(latency_target, intent, context_length):
        return get_phi3_model_config()
    elif should_use_mistral_model(intent):
        return get_mistral_model_config(intent)
    else:
        return get_default_model_config()


def get_cheap_fallback_model_config() -> tuple[str, Dict[str, Any]]:
    """Get configuration for cheap fallback model."""
    config = MODEL_CONFIGS["goblin-simple-llama-1b"]
    return config.model_id, {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "stop": config.stop_sequences,
    }


def should_use_gemma_model(
    intent: Intent,
    latency_target: LatencyTarget,
    cost_priority: bool,
    context_length: int,
) -> bool:
    """Check if Gemma 2B model should be used."""
    return (
        latency_target == LatencyTarget.ULTRA_LOW
        or intent in [Intent.CLASSIFICATION, Intent.STATUS, Intent.MICROOP]
        or (cost_priority and context_length < 100)
    )


def get_gemma_model_config() -> tuple[str, Dict[str, Any]]:
    """Get configuration for Gemma 2B model."""
    config = MODEL_CONFIGS["gemma:2b"]
    return config.model_id, {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "stop": config.stop_sequences,
    }


def should_use_qwen_model(context_length: int, language: str, intent: Intent) -> bool:
    """Check if Qwen 2.5 3B model should be used."""
    return (
        context_length > 8000
        or language != "en"
        or intent in [Intent.RAG, Intent.RETRIEVAL, Intent.TRANSLATION]
    )


def get_qwen_model_config(intent: Intent) -> tuple[str, Dict[str, Any]]:
    """Get configuration for Qwen 2.5 3B model."""
    config = MODEL_CONFIGS["qwen2.5:3b"]
    # Adjust temperature based on task
    temp = 0.0 if intent in [Intent.RAG, Intent.RETRIEVAL] else 0.3
    return config.model_id, {
        "temperature": temp,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "stop": config.stop_sequences,
    }


def should_use_phi3_model(
    latency_target: LatencyTarget, intent: Intent, context_length: int
) -> bool:
    """Check if Phi-3 3.8B model should be used."""
    return latency_target in [LatencyTarget.LOW, LatencyTarget.ULTRA_LOW] or (
        intent == Intent.CHAT and context_length < 2000
    )


def get_phi3_model_config() -> tuple[str, Dict[str, Any]]:
    """Get configuration for Phi-3 3.8B model."""
    config = MODEL_CONFIGS["phi3:3.8b"]
    return config.model_id, {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "stop": config.stop_sequences,
    }


def should_use_mistral_model(intent: Intent) -> bool:
    """Check if Mistral 7B model should be used."""
    return intent in [
        Intent.SUMMARIZE,
        Intent.EXPLAIN,
        Intent.CODE_GEN,
        Intent.CREATIVE,
        Intent.LEGAL,
    ]


def get_mistral_model_config(intent: Intent) -> tuple[str, Dict[str, Any]]:
    """Get configuration for Mistral 7B model."""
    config = MODEL_CONFIGS["mistral:7b"]
    # Adjust temperature based on intent
    if intent == Intent.CODE_GEN:
        temp = 0.0
    elif intent == Intent.CREATIVE:
        temp = 0.6
    else:
        temp = 0.2
    return config.model_id, {
        "temperature": temp,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "stop": config.stop_sequences,
    }


def get_default_model_config() -> tuple[str, Dict[str, Any]]:
    """Get configuration for default fallback model."""
    config = MODEL_CONFIGS["phi3:3.8b"]
    return config.model_id, {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "stop": config.stop_sequences,
    }


def get_system_prompt(intent: Intent) -> str:
    """Get appropriate system prompt based on intent."""
    if intent in [Intent.CODE_GEN]:
        return SYSTEM_PROMPTS["code"]
    elif intent == Intent.CREATIVE:
        return SYSTEM_PROMPTS["creative"]
    elif intent in [Intent.RAG, Intent.RETRIEVAL]:
        return SYSTEM_PROMPTS["rag"]
    elif intent in [Intent.CLASSIFICATION, Intent.STATUS]:
        return SYSTEM_PROMPTS["classification"]
    else:
        return SYSTEM_PROMPTS["default"]


def get_routing_explanation(
    model_id: str, intent: Intent, context_length: int, latency_target: LatencyTarget
) -> str:
    """Generate human-readable explanation of routing decision."""
    if model_id == "gemma:2b":
        return get_gemma_routing_explanation(intent, latency_target)
    elif model_id == "phi3:3.8b":
        return get_phi3_routing_explanation(intent, latency_target)
    elif model_id == "qwen2.5:3b":
        return get_qwen_routing_explanation(intent, context_length)
    elif model_id == "mistral:7b":
        return get_mistral_routing_explanation(intent)
    else:
        return "Default routing"


def get_gemma_routing_explanation(intent: Intent, latency_target: LatencyTarget) -> str:
    """Get routing explanation for Gemma 2B model."""
    reasons = []

    if intent in [Intent.CLASSIFICATION, Intent.STATUS, Intent.MICROOP]:
        reasons.append(f"Intent: {intent.value} (micro task)")
    if latency_target == LatencyTarget.ULTRA_LOW:
        reasons.append("Ultra-low latency required")
    reasons.append("Optimized for: ultra-fast responses, classification, status checks")

    return " | ".join(reasons)


def get_phi3_routing_explanation(intent: Intent, latency_target: LatencyTarget) -> str:
    """Get routing explanation for Phi-3 3.8B model."""
    reasons = []

    if latency_target in [LatencyTarget.LOW, LatencyTarget.ULTRA_LOW]:
        reasons.append(f"Low latency target: {latency_target.value}")
    if intent == Intent.CHAT:
        reasons.append("Conversational chat")
    reasons.append("Optimized for: low-latency chat, UI responses")

    return " | ".join(reasons)


def get_qwen_routing_explanation(intent: Intent, context_length: int) -> str:
    """Get routing explanation for Qwen 2.5 3B model."""
    reasons = []

    if context_length > 8000:
        reasons.append(f"Long context: {context_length} tokens")
    if intent in [Intent.RAG, Intent.RETRIEVAL, Intent.TRANSLATION]:
        reasons.append(f"Intent: {intent.value}")
    reasons.append("Optimized for: long documents, RAG, multilingual")

    return " | ".join(reasons)


def get_mistral_routing_explanation(intent: Intent) -> str:
    """Get routing explanation for Mistral 7B model."""
    reasons = []

    if intent in [
        Intent.SUMMARIZE,
        Intent.EXPLAIN,
        Intent.CODE_GEN,
        Intent.CREATIVE,
    ]:
        reasons.append(f"Intent: {intent.value}")
    reasons.append("Optimized for: high quality, creative, coding, explanations")

    return " | ".join(reasons)


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_cases = [
        {
            "name": "Code generation",
            "messages": [
                {"role": "user", "content": "Write a function to sort a list"}
            ],
            "intent": Intent.CODE_GEN,
        },
        {
            "name": "Quick status check",
            "messages": [{"role": "user", "content": "What's the status?"}],
            "latency_target": LatencyTarget.ULTRA_LOW,
        },
        {
            "name": "Long document RAG",
            "messages": [
                {
                    "role": "user",
                    "content": "Based on this document, what are the key points?",
                }
            ],
            "context_provided": "x" * 10000,  # 10k chars = ~2500 tokens
        },
        {
            "name": "Multilingual query",
            "messages": [{"role": "user", "content": "你好，请帮我翻译这段文字"}],
        },
        {
            "name": "Creative writing",
            "messages": [
                {"role": "user", "content": "Write a creative story about AI"}
            ],
            "intent": Intent.CREATIVE,
        },
    ]

    print("Local LLM Routing Test Cases")
    print("=" * 80)

    for test in test_cases:
        messages = test["messages"]
        intent = test.get("intent")
        latency_target = test.get("latency_target", LatencyTarget.MEDIUM)
        context_provided = test.get("context_provided")

        model_id, params = select_model(
            messages,
            intent=intent,
            latency_target=latency_target,
            context_provided=context_provided,
        )

        detected_intent = intent or detect_intent(messages)
        context_length = get_context_length(messages)
        if context_provided:
            context_length += estimate_token_count(context_provided)

        explanation = get_routing_explanation(
            model_id, detected_intent, context_length, latency_target
        )

        print(f"\nTest: {test['name']}")
        print(f"  Selected Model: {model_id}")
        print(f"  Parameters: {params}")
        print(f"  Reason: {explanation}")
        print(f"  System Prompt: {get_system_prompt(detected_intent)[:80]}...")
