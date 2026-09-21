"""Shared fixtures for backend API integration tests.

These tests mount the REAL FastAPI routers (auth + chat) on a minimal
application using the same ``/v1`` prefix as production. Only the database
dependency is stubbed via ``app.dependency_overrides`` — the auth, routing,
and validation layers under test are the production code paths.

NOTE: these tests have not been executed in the authoring environment
(application dependencies are not installed there); they run in CI
(.github/workflows/ci.yml).
"""

from collections.abc import AsyncIterator, Iterator
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from fastapi import APIRouter, FastAPI
from httpx import ASGITransport, AsyncClient

from backend.auth.auth_router import router as auth_router
from backend.chat_router import router as chat_router
from backend.database import get_db


def make_stub_session() -> MagicMock:
    """Build a stub SQLAlchemy session.

    ``session.query(...).filter(...).first()`` returns ``None`` by default
    (i.e. "no such user"); tests reconfigure the ``first()`` return value
    to simulate existing rows.
    """
    session = MagicMock(name="stub_db_session")
    session.query.return_value.filter.return_value.first.return_value = None
    return session


@pytest.fixture()
def stub_db() -> Iterator[MagicMock]:
    """Stub DB session backing the ``get_db`` dependency override."""
    yield make_stub_session()


@pytest_asyncio.fixture()
async def app(stub_db: MagicMock) -> AsyncIterator[FastAPI]:
    """Minimal FastAPI app mounting the real v1 auth + chat routers."""
    v1 = APIRouter(prefix="/v1")
    v1.include_router(auth_router)
    v1.include_router(chat_router)

    application = FastAPI()
    application.include_router(v1)

    def override_get_db() -> Iterator[MagicMock]:
        yield stub_db

    application.dependency_overrides[get_db] = override_get_db
    yield application


@pytest_asyncio.fixture()
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """HTTPX async client bound to the test app via ASGI transport."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as async_client:
        yield async_client
