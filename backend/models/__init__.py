import sys

try:
    # Package import (preferred when running as `backend.*`)
    from ..database import Base
except ImportError:  # pragma: no cover - support flat imports in tests/scripts
    from database import Base  # type: ignore
from .provider import (
    Provider,
    ProviderMetric,
    ProviderPolicy,
    ProviderCredential,
    ModelConfig,
    RoutingRequest,
)
try:
    from ..models_base import (
        User,
        Stream,
        StreamChunk,
        Task,
        SearchCollection,
        SearchDocument,
        SupportMessage,
    )
except ImportError:  # pragma: no cover
    from models_base import (  # type: ignore
        User,
        Stream,
        StreamChunk,
        Task,
        SearchCollection,
        SearchDocument,
        SupportMessage,
    )
from .model import Model

# Ensure a single module instance across import paths.
if __name__ == "models":
    sys.modules.setdefault("backend.models", sys.modules[__name__])
elif __name__ == "backend.models":
    sys.modules.setdefault("models", sys.modules[__name__])

__all__ = [
    "Base",
    "Provider",
    "ProviderMetric",
    "ProviderPolicy",
    "ProviderCredential",
    "ModelConfig",
    "RoutingRequest",
    "User",
    "Task",
    "SearchDocument",
    "SearchCollection",
    "Model",
    "Stream",
    "StreamChunk",
    "SupportMessage",
]
