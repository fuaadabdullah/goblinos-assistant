"""Integration tests for the authentication API (POST /v1/auth/login).

The real auth router is mounted; only the ``get_db`` dependency is stubbed.
"""

from unittest.mock import MagicMock

import pytest

from backend.auth.auth_router import hash_password

pytestmark = pytest.mark.asyncio


async def test_login_rejects_unknown_user(client, stub_db: MagicMock) -> None:
    """Login with an email that does not exist must return 401."""
    stub_db.query.return_value.filter.return_value.first.return_value = None

    resp = await client.post(
        "/v1/auth/login",
        json={"email": "nobody@example.com", "password": "wrong-password"},
    )

    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid credentials"


async def test_login_rejects_wrong_password(client, stub_db: MagicMock) -> None:
    """Login with a wrong password for an existing user must return 401."""
    user = MagicMock(name="existing_user")
    user.password_hash = hash_password("correct-password")
    stub_db.query.return_value.filter.return_value.first.return_value = user

    resp = await client.post(
        "/v1/auth/login",
        json={"email": "user@example.com", "password": "wrong-password"},
    )

    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid credentials"
