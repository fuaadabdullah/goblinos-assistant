from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Set

try:  # backend package import layout
    from ..providers.base import HealthStatus
except ImportError:  # backend root on PYTHONPATH
    from providers.base import HealthStatus

logger = logging.getLogger(__name__)


LEGACY_PROVIDER_ID_MAP: dict[str, str] = {
    "azure-openai": "azure_openai",
    "azure": "azure_openai",
    "ollama-gcp": "ollama_gcp",
    "llamacpp-gcp": "llamacpp_gcp",
    "openai-fallback": "openai",
    "openai_fallback": "openai",
    "openrouter-qwen": "openrouter",
    "openrouter_qwen": "openrouter",
    "aliyun-qwen": "aliyun",
    "aliyun_qwen": "aliyun",
    "aliyun-model-server": "aliyun",
    "aliyun_model_server": "aliyun",
    "alibaba": "aliyun",
    "ali-baba": "aliyun",
    "ali_baba": "aliyun",
}

_SAFE_PROVIDER_ID_RE = re.compile(r"^[a-z0-9_]+$")

_HEALTH_PRIORITY = {
    "unhealthy": 0,
    "circuit_open": 0,
    "unknown": 1,
    "degraded": 2,
    "healthy": 3,
}


def _canonical_candidate(provider_id: str) -> str:
    raw = (provider_id or "").strip().lower()
    if not raw:
        return ""
    normalized = raw.replace("-", "_").replace(" ", "_")
    return LEGACY_PROVIDER_ID_MAP.get(raw) or LEGACY_PROVIDER_ID_MAP.get(normalized) or normalized


def canonicalize_provider_id(
    provider_id: Optional[str],
    known_provider_ids: Optional[Set[str]] = None,
) -> str:
    """
    Canonicalize a provider identifier to underscore form.

    Unknown IDs follow a pass-through policy:
    - attempt canonical normalization first
    - if not safe/recognized, return the original value unchanged and log warning
    """
    raw = (provider_id or "").strip()
    if not raw:
        return ""

    candidate = _canonical_candidate(raw)
    recognized = known_provider_ids is None or candidate in known_provider_ids
    safe = bool(_SAFE_PROVIDER_ID_RE.fullmatch(candidate))
    if safe and recognized:
        return candidate

    logger.warning(
        "provider_id_unrecognized",
        extra={
            "raw_provider_id": raw,
            "canonical_candidate": candidate,
            "known_provider_ids_count": len(known_provider_ids or ()),
            "known_provider_ids_sample": sorted((known_provider_ids or set()))[:10],
        },
    )
    return raw


def build_known_provider_ids(provider_ids: Iterable[str]) -> Set[str]:
    known: Set[str] = set()
    for provider_id in provider_ids:
        candidate = _canonical_candidate(provider_id)
        if candidate and _SAFE_PROVIDER_ID_RE.fullmatch(candidate):
            known.add(candidate)
    return known


def _health_label(status: Optional[HealthStatus]) -> str:
    if status is None:
        return "unknown"
    if status == HealthStatus.HEALTHY:
        return "healthy"
    if status == HealthStatus.DEGRADED:
        return "degraded"
    return "unhealthy"


def _is_selectable(health: str) -> bool:
    return health not in {"unhealthy", "circuit_open"}


def _health_reason(health: str) -> Optional[str]:
    if health == "unhealthy":
        return "Provider health check failed."
    if health == "circuit_open":
        return "Provider temporarily blocked by circuit protection."
    return None


def _pick_health(current: str, incoming: str) -> str:
    if _HEALTH_PRIORITY[incoming] > _HEALTH_PRIORITY[current]:
        return incoming
    return current


def build_provider_index(
    registry,
    *,
    refresh_health: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Build canonical provider index for routing/model validation.

    Returns only configured/routable providers (present in registry.providers).
    """
    raw_catalog = registry.get_provider_catalog()
    configured_provider_ids = set(registry.providers.keys())
    known_ids = build_known_provider_ids(raw_catalog.keys())
    health_map = registry.get_provider_health_map(refresh=refresh_health)

    index: Dict[str, Dict[str, Any]] = {}
    for raw_provider_id, metadata in sorted(raw_catalog.items()):
        if raw_provider_id not in configured_provider_ids:
            logger.info(
                "provider_model_excluded",
                extra={
                    "provider": raw_provider_id,
                    "reason": "not_configured",
                },
            )
            continue

        canonical_provider_id = canonicalize_provider_id(raw_provider_id, known_ids)
        health = _health_label(health_map.get(raw_provider_id))
        selectable = _is_selectable(health)
        reason = _health_reason(health)
        logger.info(
            "provider_catalog_mapping",
            extra={
                "provider_key": raw_provider_id,
                "canonical_provider_id": canonical_provider_id,
                "health": health,
                "is_selectable": selectable,
                "health_reason": reason,
            },
        )

        models = metadata.get("models") or []
        normalized_models = {
            str(model_name).strip()
            for model_name in models
            if isinstance(model_name, str) and str(model_name).strip()
        }

        current = index.get(canonical_provider_id)
        if current is None:
            index[canonical_provider_id] = {
                "provider": canonical_provider_id,
                "raw_provider_ids": [raw_provider_id],
                "models": set(normalized_models),
                "health": health,
                "is_selectable": selectable,
                "health_reason": reason,
            }
            continue

        logger.info(
            "provider_catalog_collision_merge",
            extra={
                "canonical_provider_id": canonical_provider_id,
                "incoming_provider_key": raw_provider_id,
                "existing_provider_keys": list(current["raw_provider_ids"]),
            },
        )
        current["raw_provider_ids"].append(raw_provider_id)
        current["models"].update(normalized_models)
        current["health"] = _pick_health(str(current["health"]), health)
        current["is_selectable"] = bool(current["is_selectable"]) or selectable
        if not current["is_selectable"]:
            current["health_reason"] = reason
        elif current["health"] == "healthy":
            current["health_reason"] = None

    return index


def build_models_payload(registry) -> Dict[str, Any]:
    index = build_provider_index(registry, refresh_health=False)

    models: list[dict[str, Any]] = []
    providers: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for provider_id in sorted(index.keys()):
        entry = index[provider_id]
        providers.append(
            {
                "id": provider_id,
                "health": entry["health"],
                "configured": True,
                "is_selectable": bool(entry["is_selectable"]),
                "health_reason": entry.get("health_reason"),
            }
        )

        for model_name in sorted(entry["models"]):
            key = (provider_id, model_name)
            if key in seen:
                continue
            seen.add(key)
            models.append(
                {
                    "name": model_name,
                    "provider": provider_id,
                    "size": None,
                    "health": entry["health"],
                    "is_selectable": bool(entry["is_selectable"]),
                    "health_reason": entry.get("health_reason"),
                }
            )

    return {
        "models": models,
        "providers": providers,
        "source": "configured_with_health" if providers else "empty",
    }
