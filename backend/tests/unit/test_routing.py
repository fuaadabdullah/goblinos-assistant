"""
Unit tests for model routing logic.

Tests routing decisions without external API calls.

Note: After February 2026 refactoring, RoutingService uses ProviderScorer
for all scoring logic. Scorer is tested separately in test_provider_scorer.py.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from backend.services.routing_compat import RoutingServiceCompat as RoutingService


class TestRoutingService:
    """Test routing service logic."""

    @pytest.fixture
    def mock_db(self):
        """Mock database session."""
        return Mock()

    @pytest.fixture
    def routing_service(self, mock_db):
        """Create routing service with mocked dependencies."""
        # Use a valid Fernet key for testing
        test_key = "zK5sJCSNGUiCNBmYzvmzUDVemPOUZYKUdZFpJPH2hbk="
        service = RoutingService(mock_db, test_key)

        # Mock the routing manager - use Mock for sync methods
        service.routing_manager = Mock()
        # get_system_status is a sync method, so use regular return_value
        service.routing_manager.get_system_status.return_value = {
            "status": "active",
            "provider_count": 1,
            "providers": {
                "openai": {"health": "healthy", "metrics": {"latency_ms": 100}}
            },
        }
        # route_request is an async method, so use AsyncMock for it
        service.routing_manager.route_request = AsyncMock()

        return service

    @pytest.mark.asyncio
    async def test_route_request_simple_chat(self, routing_service):
        """Test routing for simple chat requests."""
        # Mock route_request result
        mock_result = Mock()
        mock_result.success = True
        mock_result.provider_id = "openai"
        mock_result.latency_ms = 100
        mock_result.cost_usd = 0.001

        routing_service.routing_manager.route_request.return_value = mock_result

        # Test routing
        result = await routing_service.route_request("chat", {"message": "Hello world"})

        assert result["success"] is True
        assert "provider" in result

    @pytest.mark.asyncio
    async def test_route_request_with_local_routing(self, routing_service):
        """Test routing that prefers local LLMs when available."""
        # Mock local LLM routing to return a result
        mock_result = Mock()
        mock_result.success = True
        mock_result.provider_id = "ollama"
        mock_result.latency_ms = 50
        mock_result.cost_usd = 0.0

        routing_service.routing_manager.route_request.return_value = mock_result

        result = await routing_service.route_request(
            "chat", {"message": "Simple question"}
        )

        assert result["success"] is True
        assert result["provider"]["id"] == "ollama"

    @pytest.mark.asyncio
    async def test_route_request_no_providers(self, routing_service):
        """Test routing when no suitable providers are available."""
        # Mock routing failure via manager exception
        routing_service.routing_manager.route_request.side_effect = Exception(
            "No providers available"
        )

        result = await routing_service.route_request("chat", {"message": "Test"})

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_discover_providers(self, routing_service):
        """Test provider discovery functionality."""
        # discover_providers uses routing_manager.get_system_status which is mocked in fixture
        result = await routing_service.discover_providers()

        assert len(result) == 1
        assert result[0]["id"] == "openai"
        assert result[0]["is_active"] is True
