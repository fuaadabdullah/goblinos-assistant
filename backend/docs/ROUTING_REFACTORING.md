# Routing Service Architecture Refactoring

**Date**: February 18, 2026
**Status**: Complete
**Goal**: Modularize routing.py to improve testability, performance, and maintainability

## Overview

The routing service has been refactored from a monolithic 800+ line file into a modular architecture with clear separation of concerns. The main routing logic now delegates to focused modules for configuration, provider management, and async database operations.

## File Structure

```
backend/services/
├── routing.py                  # Main RoutingService (now 497 lines, down from 800+)
├── provider_scorer.py          # Provider scoring module (NEW, 372 lines)
├── routing_config.py           # Configuration management (191 lines)
├── provider_registry.py        # Provider adapter initialization (271 lines)
├── async_database.py           # Async database session support (194 lines)
└── routing_helpers/            # Helper functions (pre-existing)
    ├── __init__.py
    ├── autoscaling.py
    ├── cost_analysis.py
    ├── health_metrics.py
    ├── local_llm.py
    ├── provider_selection.py
    ├── requirements.py
    └── sla_compliance.py
```

## Changes Made

### 1. Configuration Extraction (`routing_config.py`)

**Before**: Hardcoded values in `RoutingService.__init__`

```python
self.default_sla_targets = {
    "ultra_low": 500,
    "low": 1000,
    "medium": 2000,
    "high": 5000,
}
self.cost_budget_weights = {...}
self.adapters = {...}  # 11 hardcoded adapters
```

**After**: Centralized configuration with environment variable support

```python
config = RoutingConfig.from_env()
# or
config = RoutingConfig(
    sla_targets=custom_sla_targets,
    cost_budget_weights=custom_weights,
    adapter_registry=custom_adapters,
)
service = RoutingService(db=db, config=config)
```

**Benefits**:

- Testable: Inject custom config for unit tests
- Configurable: Override via environment variables
- Validated: Built-in config validation
- Documented: Single source of truth for defaults

**Environment Variables**:

```bash
# SLA Targets (milliseconds)
SLA_ULTRA_LOW_MS=500
SLA_LOW_MS=1000
SLA_MEDIUM_MS=2000
SLA_HIGH_MS=5000

# Scoring weights (must sum to ~1.0)
WEIGHT_LATENCY=0.3
WEIGHT_COST=0.4
WEIGHT_SLA=0.3

# Required for API key decryption
ROUTING_ENCRYPTION_KEY=your-32-character-key
```

### 2. Async Database Support (`async_database.py`)

**Before**: Sync queries wrapped in `asyncio.to_thread()`

```python
async def discover_providers(self):
    def _sync_query():
        return self.db.query(Provider).filter(Provider.is_active).all()
    providers = await asyncio.to_thread(_sync_query)  # Thread pool overhead!
```

**After**: Direct async queries with AsyncSession

```python
async def discover_providers(self):
    if self.async_db:
        result = await self.async_db.execute(
            select(Provider).where(Provider.is_active == True)
        )
        providers = result.scalars().all()  # Pure async, no threads!
```

**Benefits**:

- Performance: Eliminates thread pool overhead on critical path
- Scalability: Better connection pooling for high concurrency
- Modern: Uses SQLAlchemy 2.0 async patterns
- Backward compatible: Falls back to sync session if provided

**New Dependency**:

```bash
pip install asyncpg>=0.30.0  # PostgreSQL async driver
```

**Database URL**: Automatically converts `postgresql://` to `postgresql+asyncpg://`

### 3. Provider Registry (`provider_registry.py`)

**Before**: 80+ lines of adapter initialization in `discover_providers()`

```python
# Get API key - try encrypted first, fall back to plain text
api_key = None
if provider.api_key_encrypted:
    try:
        api_key = self.encryption_service.decrypt(provider.api_key_encrypted)
    except Exception as e:
        logger.warning(f"Failed to decrypt: {e}")
# ... 50 more lines ...
adapter = adapter_class(api_key, base_url)
models = await adapter.list_models()
```

**After**: Delegated to `ProviderRegistry`

```python
adapter = await self.provider_registry.initialize_adapter(
    provider_name=provider.name,
    encrypted_key=provider.api_key_encrypted,
    plain_key=provider.api_key,
    base_url=provider.base_url,
)
models = await self.provider_registry.get_provider_models(adapter)
```

**Benefits**:

- Single responsibility: Registry handles all adapter concerns
- Reusable: Can be used outside routing service
- Testable: Mock registry instead of 11 adapters
- Maintainable: Add new providers in one place

**API**:

```python
registry = ProviderRegistry(
    adapter_registry=config.adapter_registry,
    encryption_service=encryption_service,
)

# Initialize adapter with automatic key resolution
adapter = await registry.initialize_adapter("openai", encrypted_key=key)

# Get models
models = await registry.get_provider_models(adapter)
```

### 4. Provider Scoring Module (`provider_scorer.py`)

**Date**: February 18, 2026 (Phase 2 refactoring)
**Lines extracted**: ~200 lines from `routing.py`

**Before**: All scoring logic embedded in `RoutingService`

```python
class RoutingService:
    async def _score_providers(self, providers, capability, ...):
        scored = []
        for provider in providers:
            score = await self._calculate_provider_score(...)
            # 100+ lines of scoring logic
        return scored

    async def _calculate_provider_score(self, ...):
        # Multi-factor scoring algorithm
        base_score = 50.0
        health_score = await self._get_health_score(...)
        # ... 80 more lines ...
        return score

    # + 7 more scoring methods
```

**After**: Dedicated `ProviderScorer` class

```python
from backend.services.provider_scorer import ProviderScorer

class RoutingService:
    def __init__(self, ...):
        # Initialize scorer
        self.scorer = ProviderScorer(
            config=config,
            db=self.db,
            async_db=self.async_db,
            latency_monitor=self.latency_monitor,
        )

    async def _score_providers(self, providers, capability, ...):
        # Delegate to scorer
        return await self.scorer.score_providers(
            providers=providers,
            capability=capability,
            requirements=requirements,
            sla_target_ms=sla_target_ms,
            cost_budget=cost_budget,
            latency_priority=latency_priority,
        )
```

**Benefits**:

- **Modularity**: Scoring logic isolated from routing logic
- **Testability**: Score providers independently without routing overhead
- **Maintainability**: Single module to modify for scoring changes
- **Reusability**: Can use scorer in other contexts (batch evaluation, monitoring)
- **Clarity**: ~200 lines removed from routing.py (28% reduction)

**Scoring Algorithm** (documented in scorer module):

```
Base score: 50.0 (neutral)
+ Health score: -50 to +75 (weighted by latency priority)
+ Priority bonus: 0 to +20 (provider tier * 2.0)
+ SLA compliance: -20 to +20 (weighted)
+ Performance bonus: 0 to +15 (weighted by latency priority)
+ Capability bonus: 0 to +10 (feature matching)
- Cost penalty: 0 to -30 (weighted by cost priority)
= Final score: 0 to 100 (clamped)
```

**Key Factors**:

1. **Health Metrics**: Error rates, availability, recent failures
2. **Performance**: Latency P50/P95/P99, throughput
3. **Cost**: Per-token pricing, budget constraints
4. **SLA Compliance**: Response time vs targets
5. **Capability Match**: Model features, context length
6. **Latency Priority**: User-specified performance needs

**Testing**:

```bash
# Test scorer independently
pytest backend/tests/unit/test_provider_scorer.py -v

# Test with various scoring scenarios
pytest backend/tests/unit/test_provider_scorer.py::TestScoreProviders -v
```

**API Example**:

```python
# Direct usage (for testing/monitoring)
scorer = ProviderScorer(config, db=session)
providers = [...]

scored = await scorer.score_providers(
    providers=providers,
    capability="chat",
    sla_target_ms=1000.0,
    cost_budget=0.01,
    latency_priority="low",
)

print(f"Best provider: {scored[0]['name']} (score: {scored[0]['score']})")
```

## Migration Guide

### For Existing Code

**Old initialization**:

```python
from backend.services.routing import RoutingService
from backend.database import SessionLocal

db = SessionLocal()
service = RoutingService(db, encryption_key="your-key")
```

**New initialization (backward compatible)**:

```python
from backend.services.routing import RoutingService
from backend.database import SessionLocal

db = SessionLocal()
service = RoutingService(db=db, encryption_key="your-key")  # Still works!
```

**New initialization (recommended)**:

```python
from backend.services.routing import RoutingService
from backend.services.routing_config import RoutingConfig
from backend.async_database import get_async_db

# Use async session for better performance
async for async_db in get_async_db():
    config = RoutingConfig.from_env()
    service = RoutingService(async_db=async_db, config=config)

    # Use service
    providers = await service.discover_providers()
```

### For Tests

**Old test setup**:

```python
def test_routing():
    mock_db = MagicMock(spec=Session)
    service = RoutingService(mock_db, "test-key")
```

**New test setup**:

```python
import pytest
from backend.services.routing_config import RoutingConfig

@pytest.fixture
def test_config():
    return RoutingConfig(
        sla_targets={"ultra_low": 100, "low": 200, "medium": 300, "high": 400},
        cost_budget_weights={"latency_priority": 0.33, "cost_priority": 0.33, "sla_compliance": 0.34},
        adapter_registry={"test": MockAdapter},
        encryption_key="test-key-32-characters-long-123",
    )

async def test_routing(test_config):
    mock_db = AsyncMock(spec=AsyncSession)
    service = RoutingService(async_db=mock_db, config=test_config)

    # Test async methods
    providers = await service.discover_providers()
```

## Performance Impact

### Before

- **Thread pool overhead**: Every DB query spawned a thread
- **Connection pooling**: Limited by sync engine constraints
- **Latency**: ~5-15ms overhead per query from threading

### After

- **Pure async**: No thread pool, direct async I/O
- **Better pooling**: AsyncEngine supports larger connection pools
- **Latency**: <1ms overhead, only I/O wait time

**Estimated improvement**: 10-50ms faster on routing critical path (depends on query count)

## Testing

### Unit Tests

```bash
# Test configuration module
pytest backend/test_routing_config.py -v

# Test routing service with mocked async DB
pytest backend/tests/test_routing.py -v
```

### Integration Tests

```bash
# Requires PostgreSQL with DATABASE_URL set
pytest backend/test_routing_endpoints.py -v

# With async database
DATABASE_URL=postgresql+asyncpg://user:pass@host/db pytest backend/tests/integration/
```

## Known Issues & Limitations

1. **Routing helpers still use sync DB**: The `routing_helpers/` modules still use `asyncio.to_thread()` for DB queries. Migrating these is a future improvement.

2. **Backward compatibility shims**: Both sync and async sessions are supported, adding some complexity. Consider deprecating sync session in a future release.

3. **SQLite not supported for async**: The async database module requires PostgreSQL. Local development should use Supabase or Docker Postgres.

4. **Scorer still supports sync sessions**: For backward compatibility, `ProviderScorer` accepts both sync and async database sessions. Future versions will require async only.

## Future Improvements

1. **Migrate routing_helpers to async**: Convert `health_metrics.py`, `sla_compliance.py` to use AsyncSession
2. **Extract emergency routing**: Move emergency/fallback routing to dedicated module (~60 lines)
3. **Deprecate sync session**: Remove `db` parameter in favor of `async_db` only
4. **Configuration validation**: Add Pydantic models for stricter validation
5. **Provider registry caching**: Cache initialized adapters to reduce initialization overhead
6. **Scorer optimization**: Batch database queries for metrics to reduce query count

## Rollback Plan

If issues arise, revert to previous version:

```bash
git revert <commit-hash>
```

The refactoring maintains backward compatibility, so rolling back should be safe.

## Questions & Support

- Documentation: See inline docstrings in `routing_config.py`, `provider_registry.py`, `async_database.py`
- Issues: File tickets in Jira with tag `routing-refactoring`
- Contact: Backend team

---

**Last updated**: February 18, 2026
