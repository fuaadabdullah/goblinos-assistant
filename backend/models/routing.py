# This file has been consolidated into apps/goblin-assistant-root/backend/models/provider.py
# on 2025-12-11. Please refer to that file for the updated Provider, ProviderMetric,
# ProviderPolicy, ProviderCredential, ModelConfig, and RoutingRequest models.

# Backwards-compatible re-exports for any code still importing from models.routing
from .provider import Provider as RoutingProvider  # noqa: F401
from .provider import ProviderMetric  # noqa: F401
from .provider import RoutingRequest  # noqa: F401
