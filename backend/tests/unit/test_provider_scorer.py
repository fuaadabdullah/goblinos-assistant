"""
Unit tests for ProviderScorer module.

Tests cover provider scoring logic isolated from routing service:
- Basic scoring with various factors
- Health score calculations
- Cost penalties and budget constraints
- SLA compliance scoring
- Capability matching bonuses
- Latency priority weights
- Edge cases and error handling
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from backend.services.provider_scorer import ProviderScorer
from backend.services.routing_config import RoutingConfig


@pytest.fixture
def mock_config():
    """Create a mock RoutingConfig for testing."""
    config = Mock(spec=RoutingConfig)
    config.cost_budget_weights = {
        "latency_priority": 1.0,
        "sla_compliance": 1.0,
        "cost_priority": 1.0,
    }
    config.sla_targets = {
        "chat": 1000.0,
        "completion": 2000.0,
    }
    return config


@pytest.fixture
def mock_db():
    """Create a mock database session."""
    return Mock()


@pytest.fixture
def mock_async_db():
    """Create a mock async database session."""
    return AsyncMock()


@pytest.fixture
def mock_latency_monitor():
    """Create a mock latency monitoring service."""
    return Mock()


@pytest.fixture
def scorer(mock_config, mock_db, mock_latency_monitor):
    """Create a ProviderScorer instance for testing."""
    return ProviderScorer(
        config=mock_config,
        db=mock_db,
        latency_monitor=mock_latency_monitor,
    )


@pytest.fixture
def sample_provider():
    """Create a sample provider dict for testing."""
    return {
        "id": 1,
        "name": "TestProvider",
        "priority": 5,
        "capabilities": ["chat", "completion"],
        "pricing": {"chat": 0.002, "completion": 0.001},
    }


@pytest.fixture
def sample_providers():
    """Create a list of sample providers for testing."""
    return [
        {
            "id": 1,
            "name": "FastProvider",
            "priority": 8,
            "capabilities": ["chat", "completion"],
        },
        {
            "id": 2,
            "name": "CheapProvider",
            "priority": 3,
            "capabilities": ["chat"],
        },
        {
            "id": 3,
            "name": "BalancedProvider",
            "priority": 5,
            "capabilities": ["chat", "completion", "embeddings"],
        },
    ]


class TestProviderScorerInit:
    """Test scorer initialization."""

    def test_init_with_sync_db(self, mock_config, mock_db, mock_latency_monitor):
        """Test initialization with sync database session."""
        scorer = ProviderScorer(
            config=mock_config,
            db=mock_db,
            latency_monitor=mock_latency_monitor,
        )
        assert scorer.db == mock_db
        assert scorer.async_db is None
        assert scorer.config == mock_config

    def test_init_with_async_db(self, mock_config, mock_async_db, mock_latency_monitor):
        """Test initialization with async database session."""
        scorer = ProviderScorer(
            config=mock_config,
            async_db=mock_async_db,
            latency_monitor=mock_latency_monitor,
        )
        assert scorer.async_db == mock_async_db
        assert scorer.db is None

    def test_init_with_both_db_prefers_async(
        self, mock_config, mock_db, mock_async_db, mock_latency_monitor
    ):
        """Test that async_db takes precedence when both provided."""
        scorer = ProviderScorer(
            config=mock_config,
            db=mock_db,
            async_db=mock_async_db,
            latency_monitor=mock_latency_monitor,
        )
        assert scorer.async_db == mock_async_db
        assert scorer.db is None

    def test_init_without_db_raises_error(self, mock_config, mock_latency_monitor):
        """Test that initialization without db raises ValueError."""
        with pytest.raises(ValueError, match="Either db or async_db must be provided"):
            ProviderScorer(
                config=mock_config,
                latency_monitor=mock_latency_monitor,
            )

    def test_init_creates_latency_monitor_if_none(self, mock_config, mock_db):
        """Test that latency monitor is created if not provided."""
        scorer = ProviderScorer(config=mock_config, db=mock_db)
        assert scorer.latency_monitor is not None


class TestScoreProviders:
    """Test the main score_providers method."""

    @pytest.mark.asyncio
    async def test_score_providers_basic(self, scorer, sample_providers):
        """Test basic provider scoring."""
        with patch.object(scorer, "_calculate_provider_score", return_value=75.0):
            scored = await scorer.score_providers(
                providers=sample_providers,
                capability="chat",
            )

            assert len(scored) == 3
            assert all("score" in p for p in scored)
            assert all(p["score"] == 75.0 for p in scored)

    @pytest.mark.asyncio
    async def test_score_providers_sorted_descending(self, scorer, sample_providers):
        """Test that providers are sorted by score descending."""
        scores = [85.0, 60.0, 95.0]  # Not in order

        async def mock_score(provider, *args, **kwargs):
            return scores[sample_providers.index(provider)]

        with patch.object(scorer, "_calculate_provider_score", side_effect=mock_score):
            scored = await scorer.score_providers(
                providers=sample_providers,
                capability="chat",
            )

            # Should be sorted: 95, 85, 60
            assert scored[0]["id"] == 3  # BalancedProvider (score 95)
            assert scored[1]["id"] == 1  # FastProvider (score 85)
            assert scored[2]["id"] == 2  # CheapProvider (score 60)

    @pytest.mark.asyncio
    async def test_score_providers_filters_zero_scores(self, scorer, sample_providers):
        """Test that providers with zero or negative scores are filtered out."""
        scores = [75.0, 0.0, -10.0]

        async def mock_score(provider, *args, **kwargs):
            return scores[sample_providers.index(provider)]

        with patch.object(scorer, "_calculate_provider_score", side_effect=mock_score):
            scored = await scorer.score_providers(
                providers=sample_providers,
                capability="chat",
            )

            assert len(scored) == 1
            assert scored[0]["id"] == 1  # Only FastProvider (75.0)

    @pytest.mark.asyncio
    async def test_score_providers_empty_list(self, scorer):
        """Test scoring with empty provider list."""
        scored = await scorer.score_providers(
            providers=[],
            capability="chat",
        )
        assert scored == []

    @pytest.mark.asyncio
    async def test_score_providers_with_all_parameters(self, scorer, sample_providers):
        """Test scoring with all optional parameters."""
        with patch.object(scorer, "_calculate_provider_score", return_value=80.0):
            scored = await scorer.score_providers(
                providers=sample_providers,
                capability="chat",
                requirements={"context_length": 8000},
                sla_target_ms=1000.0,
                cost_budget=0.01,
                latency_priority="low",
            )

            assert len(scored) == 3
            # Verify calculate_provider_score was called with all params
            scorer._calculate_provider_score.assert_called()


class TestCalculateProviderScore:
    """Test individual provider score calculation."""

    @pytest.mark.asyncio
    async def test_base_score_starts_at_50(self, scorer, sample_provider):
        """Test that base score starts at 50."""
        with (
            patch.object(scorer, "_get_health_score", return_value=0.0),
            patch.object(scorer, "_get_performance_bonus", return_value=0.0),
            patch.object(scorer, "_calculate_capability_bonus", return_value=0.0),
            patch.object(
                scorer, "_calculate_cost_penalty_with_budget", return_value=0.0
            ),
        ):
            score = await scorer._calculate_provider_score(
                provider=sample_provider,
                capability="chat",
            )
            # Base (50) + priority bonus (5 * 2 = 10) = 60
            assert score == 60.0

    @pytest.mark.asyncio
    async def test_score_includes_health_bonus(self, scorer, sample_provider):
        """Test that health score contributes to total."""
        with (
            patch.object(scorer, "_get_health_score", return_value=20.0),
            patch.object(scorer, "_get_performance_bonus", return_value=0.0),
            patch.object(scorer, "_calculate_capability_bonus", return_value=0.0),
            patch.object(
                scorer, "_calculate_cost_penalty_with_budget", return_value=0.0
            ),
        ):
            score = await scorer._calculate_provider_score(
                provider=sample_provider,
                capability="chat",
            )
            # Base (50) + health (20 * 1.0 weight) + priority (10) = 80
            assert score == 80.0

    @pytest.mark.asyncio
    async def test_score_includes_cost_penalty(self, scorer, sample_provider):
        """Test that cost penalties reduce score."""
        with (
            patch.object(scorer, "_get_health_score", return_value=0.0),
            patch.object(scorer, "_get_performance_bonus", return_value=0.0),
            patch.object(scorer, "_calculate_capability_bonus", return_value=0.0),
            patch.object(
                scorer, "_calculate_cost_penalty_with_budget", return_value=15.0
            ),
        ):
            score = await scorer._calculate_provider_score(
                provider=sample_provider,
                capability="chat",
            )
            # Base (50) + priority (10) - cost penalty (15 * 1.0 weight) = 45
            assert score == 45.0

    @pytest.mark.asyncio
    async def test_score_clamped_to_100(self, scorer, sample_provider):
        """Test that scores are clamped to maximum of 100."""
        with (
            patch.object(scorer, "_get_health_score", return_value=75.0),
            patch.object(scorer, "_get_performance_bonus", return_value=15.0),
            patch.object(scorer, "_calculate_capability_bonus", return_value=10.0),
            patch.object(
                scorer, "_calculate_cost_penalty_with_budget", return_value=0.0
            ),
        ):
            score = await scorer._calculate_provider_score(
                provider=sample_provider,
                capability="chat",
            )
            # Would be 50 + 75 + 10 + 15 + 10 = 160, clamped to 100
            assert score == 100.0

    @pytest.mark.asyncio
    async def test_score_clamped_to_0(self, scorer, sample_provider):
        """Test that scores are clamped to minimum of 0."""
        with (
            patch.object(scorer, "_get_health_score", return_value=-50.0),
            patch.object(scorer, "_get_performance_bonus", return_value=0.0),
            patch.object(scorer, "_calculate_capability_bonus", return_value=0.0),
            patch.object(
                scorer, "_calculate_cost_penalty_with_budget", return_value=30.0
            ),
        ):
            score = await scorer._calculate_provider_score(
                provider=sample_provider,
                capability="chat",
            )
            # 50 - 50 + 10 - 30 = -20, clamped to 0
            assert score == 0.0

    @pytest.mark.asyncio
    async def test_score_with_sla_target(self, scorer, sample_provider):
        """Test that SLA target is considered in scoring."""
        with (
            patch.object(scorer, "_get_health_score", return_value=0.0),
            patch.object(scorer, "_get_performance_bonus", return_value=0.0),
            patch.object(scorer, "_calculate_capability_bonus", return_value=0.0),
            patch.object(
                scorer, "_calculate_cost_penalty_with_budget", return_value=0.0
            ),
            patch.object(scorer, "_calculate_sla_score", return_value=15.0),
        ):
            score = await scorer._calculate_provider_score(
                provider=sample_provider,
                capability="chat",
                sla_target_ms=1000.0,
            )
            # Base (50) + priority (10) + SLA (15 * 1.0 weight) = 75
            assert score == 75.0

    @pytest.mark.asyncio
    async def test_score_with_latency_priority(self, scorer, sample_provider):
        """Test that latency priority affects performance weighting."""
        with (
            patch.object(scorer, "_get_health_score", return_value=0.0),
            patch.object(scorer, "_get_performance_bonus", return_value=10.0),
            patch.object(scorer, "_calculate_capability_bonus", return_value=0.0),
            patch.object(
                scorer, "_calculate_cost_penalty_with_budget", return_value=0.0
            ),
        ):
            score = await scorer._calculate_provider_score(
                provider=sample_provider,
                capability="chat",
                latency_priority="ultra_low",
            )
            # Base (50) + priority (10) + performance (10 * 2.0 weight ultra_low) = 80
            assert score == 80.0


class TestGetLatencyWeight:
    """Test latency weight calculation."""

    def test_ultra_low_latency_weight(self, scorer):
        """Test ultra_low latency priority returns 2.0."""
        assert scorer._get_latency_weight("ultra_low") == 2.0

    def test_low_latency_weight(self, scorer):
        """Test low latency priority returns 1.5."""
        assert scorer._get_latency_weight("low") == 1.5

    def test_medium_latency_weight(self, scorer):
        """Test medium latency priority returns 1.0."""
        assert scorer._get_latency_weight("medium") == 1.0

    def test_high_latency_weight(self, scorer):
        """Test high latency priority returns 0.7."""
        assert scorer._get_latency_weight("high") == 0.7

    def test_unknown_latency_weight_defaults_to_1(self, scorer):
        """Test unknown latency priority defaults to 1.0."""
        assert scorer._get_latency_weight("unknown") == 1.0
        assert scorer._get_latency_weight("") == 1.0
        assert scorer._get_latency_weight(None) == 1.0


class TestHelperMethods:
    """Test that helper methods delegate to routing_helpers correctly."""

    @pytest.mark.asyncio
    async def test_get_health_score_calls_helper(self, scorer):
        """Test that _get_health_score delegates to routing_helpers."""
        with patch(
            "backend.services.routing_helpers.calculate_provider_health_score"
        ) as mock:
            mock.return_value = 50.0
            result = await scorer._get_health_score(provider_id=1)
            assert result == 50.0
            mock.assert_called_once_with(scorer.db, 1)

    def test_calculate_cost_penalty_calls_helper(self, scorer, sample_provider):
        """Test that _calculate_cost_penalty delegates to routing_helpers."""
        with patch("backend.services.routing_helpers.calculate_cost_penalty") as mock:
            mock.return_value = 10.0
            result = scorer._calculate_cost_penalty(sample_provider, "chat")
            assert result == 10.0
            mock.assert_called_once_with(sample_provider, "chat")

    @pytest.mark.asyncio
    async def test_get_performance_bonus_calls_helper(self, scorer):
        """Test that _get_performance_bonus delegates to routing_helpers."""
        with patch(
            "backend.services.routing_helpers.calculate_provider_performance_bonus"
        ) as mock:
            mock.return_value = 12.0
            result = await scorer._get_performance_bonus(provider_id=1)
            assert result == 12.0
            mock.assert_called_once_with(scorer.db, 1)

    def test_calculate_capability_bonus_calls_helper(self, scorer, sample_provider):
        """Test that _calculate_capability_bonus delegates to routing_helpers."""
        with patch(
            "backend.services.routing_helpers.calculate_capability_bonus"
        ) as mock:
            mock.return_value = 8.0
            requirements = {"context_length": 8000}
            result = scorer._calculate_capability_bonus(
                sample_provider, "chat", requirements
            )
            assert result == 8.0
            mock.assert_called_once_with(sample_provider, "chat", requirements)

    @pytest.mark.asyncio
    async def test_calculate_sla_score_calls_helper(self, scorer, sample_provider):
        """Test that _calculate_sla_score delegates to routing_helpers."""
        with patch(
            "backend.services.routing_helpers.calculate_provider_sla_score"
        ) as mock:
            mock.return_value = 15.0
            result = await scorer._calculate_sla_score(
                sample_provider, sla_target_ms=1000.0
            )
            assert result == 15.0
            mock.assert_called_once()

    def test_calculate_cost_penalty_with_budget_calls_helper(
        self, scorer, sample_provider
    ):
        """Test that _calculate_cost_penalty_with_budget delegates to routing_helpers."""
        with patch(
            "backend.services.routing_helpers.calculate_cost_penalty_with_budget"
        ) as mock:
            mock.return_value = 20.0
            result = scorer._calculate_cost_penalty_with_budget(
                sample_provider, "chat", cost_budget=0.01
            )
            assert result == 20.0
            mock.assert_called_once_with(sample_provider, "chat", 0.01)


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_provider_without_priority(self, scorer):
        """Test handling provider dict without priority field."""
        provider = {"id": 1, "name": "NoPriority"}
        with (
            patch.object(scorer, "_get_health_score", return_value=0.0),
            patch.object(scorer, "_get_performance_bonus", return_value=0.0),
            patch.object(scorer, "_calculate_capability_bonus", return_value=0.0),
            patch.object(
                scorer, "_calculate_cost_penalty_with_budget", return_value=0.0
            ),
        ):
            score = await scorer._calculate_provider_score(provider, "chat")
            # Should handle missing priority gracefully (defaults to 0)
            assert score == 50.0  # Base score only

    @pytest.mark.asyncio
    async def test_score_providers_preserves_original_dicts(
        self, scorer, sample_providers
    ):
        """Test that original provider dicts are not mutated."""
        original_providers = [p.copy() for p in sample_providers]

        with patch.object(scorer, "_calculate_provider_score", return_value=75.0):
            scored = await scorer.score_providers(
                providers=sample_providers,
                capability="chat",
            )

            # Original providers should be unchanged
            for orig, current in zip(original_providers, sample_providers):
                assert orig == current
                assert "score" not in current

            # Scored providers should have scores
            assert all("score" in p for p in scored)
