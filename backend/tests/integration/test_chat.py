"""Integration tests for the chat API auth boundary.

The real chat router is mounted; only the ``get_db`` dependency is stubbed.
Chat-service construction is additionally stubbed so the test exercises
exactly the authentication layer (no provider or DB work happens).
"""

from unittest.mock import MagicMock

import pytest

from backend.chat_router import get_chat_service

pytestmark = pytest.mark.asyncio


async def test_chat_completions_rejects_unauthenticated_request(client, app) -> None:
    """POST /v1/chat/completions without credentials must be rejected."""
    # Focus the test on the auth layer: FastAPI's HTTPBearer security
    # dependency rejects the request before any chat-service work happens.
    app.dependency_overrides[get_chat_service] = lambda: MagicMock(
        name="stub_chat_service"
    )

    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    # HTTPBearer raises 403 when the Authorization header is missing;
    # a present-but-invalid token surfaces as 401 from require_scope.
    assert resp.status_code in (401, 403)
