import logging

from fastapi import APIRouter, Request, Depends

from ..services.imports import get_chat_routing_service
from ..errors import raise_internal_error

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.get("/models")
async def list_available_models(
    req: Request,
    service=Depends(get_chat_routing_service),
):
    """
    List all available models across all providers.
    Includes routing recommendations for each model.
    """
    try:
        providers = await service.discover_providers()

        models = []
        for provider in providers:
            for model in provider["models"]:
                models.append(
                    {
                        "id": model["id"],
                        "provider": provider["display_name"],
                        "provider_name": provider["name"],
                        "capabilities": model.get("capabilities", []),
                        "context_window": model.get("context_window", 0),
                        "pricing": model.get("pricing", {}),
                    }
                )

        # Add routing recommendations
        routing_recommendations = {
            "gemma:2b": "Ultra-fast responses, classification, status checks",
            "phi3:3.8b": "Low-latency chat, conversational UI",
            "qwen2.5:3b": "Long context (32K), multilingual, RAG",
            "mistral:7b": "High quality, code generation, creative writing",
        }

        for model in models:
            model["routing_recommendation"] = routing_recommendations.get(
                model["id"], "General purpose"
            )

        return {
            "models": models,
            "total_count": len(models),
            "routing_info": {
                "automatic": True,
                "factors": [
                    "intent",
                    "context_length",
                    "latency_target",
                    "cost_priority",
                ],
                "documentation": "/docs/LOCAL_LLM_ROUTING.md",
            },
        }

    except Exception as exc:
        logger.error(
            "Failed to list models",
            extra={
                "error": str(exc),
                "correlation_id": getattr(req.state, "correlation_id", None),
                "request_id": getattr(req.state, "request_id", None),
            },
        )
        raise_internal_error(f"Failed to list models: {str(exc)}")


@router.get("/routing-info")
async def get_routing_info():
    """
    Get information about the intelligent routing system.
    """
    return {
        "routing_system": "intelligent",
        "version": "1.0",
        "factors": {
            "intent": {
                "description": "Detected or explicit intent (code-gen, creative, rag, chat, etc.)",
                "options": [
                    "code-gen",
                    "creative",
                    "explain",
                    "summarize",
                    "rag",
                    "retrieval",
                    "chat",
                    "classification",
                    "status",
                    "translation",
                ],
                "auto_detect": True,
            },
            "latency_target": {
                "description": "Target latency for response",
                "options": ["ultra_low", "low", "medium", "high"],
                "default": "medium",
            },
            "context_length": {
                "description": "Length of the conversation context",
                "thresholds": {
                    "short": "< 2000 tokens",
                    "medium": "2000-8000 tokens",
                    "long": "> 8000 tokens (uses qwen2.5:3b with 32K window)",
                },
            },
            "cost_priority": {
                "description": "Prioritize cost over quality",
                "default": False,
                "effect": "Routes to smaller, faster models when enabled",
            },
        },
        "models": {
            "gemma:2b": {
                "size": "1.7GB",
                "context": "8K tokens",
                "latency": "5-8s",
                "best_for": ["ultra_fast", "classification", "status_checks"],
                "params": {"temperature": 0.0, "max_tokens": 40},
            },
            "phi3:3.8b": {
                "size": "2.2GB",
                "context": "4K tokens",
                "latency": "10-12s",
                "best_for": ["low_latency_chat", "conversational_ui", "quick_qa"],
                "params": {"temperature": 0.15, "max_tokens": 128},
            },
            "qwen2.5:3b": {
                "size": "1.9GB",
                "context": "32K tokens",
                "latency": "14s",
                "best_for": [
                    "long_context",
                    "multilingual",
                    "rag",
                    "document_retrieval",
                ],
                "params": {"temperature": 0.0, "max_tokens": 1024},
            },
            "mistral:7b": {
                "size": "4.4GB",
                "context": "8K tokens",
                "latency": "14-15s",
                "best_for": [
                    "high_quality",
                    "code_generation",
                    "creative_writing",
                    "explanations",
                ],
                "params": {"temperature": 0.2, "max_tokens": 512},
            },
        },
        "cost": {
            "per_request": "$0 (self-hosted)",
            "monthly_infrastructure": "$15-20",
            "savings_vs_cloud": "86-92% ($110-240/month)",
        },
    }


__all__ = ["router"]
