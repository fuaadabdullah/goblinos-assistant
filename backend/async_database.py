"""
Async database session support for routing service and other async operations.

This module provides async SQLAlchemy session support to eliminate the need for
asyncio.to_thread() wrappers in async code paths. It complements the existing
synchronous database.py module for backward compatibility.
"""

import os
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.pool import NullPool, QueuePool
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()


def _get_async_database_url() -> str:
    """Get async database URL from environment.

    Converts postgresql:// URLs to postgresql+asyncpg:// for async support.

    Returns:
        Async-compatible database URL

    Raises:
        ValueError: If DATABASE_URL not set or is SQLite (not supported for async)
    """
    db_url = os.getenv("DATABASE_URL", "")

    if not db_url:
        raise ValueError(
            "DATABASE_URL environment variable not set. "
            "Async database operations require PostgreSQL."
        )

    # Check if it's SQLite (not supported for async operations)
    if db_url.startswith("sqlite"):
        raise ValueError(
            "SQLite is not supported for async database operations. "
            "Please use PostgreSQL (Supabase) for production."
        )

    # Convert postgres:// to postgresql+asyncpg:// for async support
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif not db_url.startswith("postgresql+asyncpg://"):
        # Already has asyncpg or different scheme
        logger.warning(
            f"Database URL has unexpected scheme: {db_url[:20]}... "
            "Proceeding anyway, but connection may fail."
        )

    return db_url


def create_async_engine_instance() -> AsyncEngine:
    """Create async SQLAlchemy engine with production-ready configuration.

    Returns:
        Configured AsyncEngine instance

    Raises:
        ValueError: If database URL is invalid or not set
    """
    database_url = _get_async_database_url()

    # Detect Supabase for connection pool tuning
    is_supabase = "supabase" in database_url or "pooler.supabase" in database_url

    # Connection pool configuration
    # Supabase uses Supavisor connection pooler, so we use smaller local pools
    pool_config = {
        "pool_size": int(os.getenv("ASYNC_DB_POOL_SIZE", "5" if is_supabase else "20")),
        "max_overflow": int(
            os.getenv("ASYNC_DB_MAX_OVERFLOW", "5" if is_supabase else "10")
        ),
        "pool_timeout": int(os.getenv("ASYNC_DB_POOL_TIMEOUT", "30")),
        "pool_recycle": int(
            os.getenv("ASYNC_DB_POOL_RECYCLE", "300" if is_supabase else "3600")
        ),
        "pool_pre_ping": True,
        "echo": False,
    }

    # Use QueuePool for production, NullPool for testing
    if os.getenv("TESTING") == "true":
        pool_config["poolclass"] = NullPool
    else:
        pool_config["poolclass"] = QueuePool

    # Connection arguments (asyncpg specific)
    connect_args = {
        "timeout": 10,  # Connection timeout
        "command_timeout": 30,  # Statement timeout (30s)
    }

    # Supabase requires SSL
    if is_supabase:
        connect_args["ssl"] = "require"

    # Create async engine
    engine = create_async_engine(
        database_url,
        connect_args=connect_args,
        **pool_config,
    )

    logger.info(
        f"Created async database engine (pool_size={pool_config['pool_size']}, "
        f"max_overflow={pool_config['max_overflow']})"
    )

    return engine


# Global async engine and session maker
_async_engine: Optional[AsyncEngine] = None
_async_session_maker: Optional[async_sessionmaker] = None


def get_async_engine() -> AsyncEngine:
    """Get or create the global async database engine.

    Returns:
        AsyncEngine instance singleton
    """
    global _async_engine
    if _async_engine is None:
        _async_engine = create_async_engine_instance()
    return _async_engine


def get_async_session_maker() -> async_sessionmaker:
    """Get or create the global async session maker.

    Returns:
        async_sessionmaker factory singleton
    """
    global _async_session_maker
    if _async_session_maker is None:
        engine = get_async_engine()
        _async_session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    return _async_session_maker


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency to get async database session.

    Usage:
        @app.get("/endpoint")
        async def endpoint(db: AsyncSession = Depends(get_async_db)):
            result = await db.execute(select(Model))
            return result.scalars().all()

    Yields:
        AsyncSession instance
    """
    session_maker = get_async_session_maker()
    async with session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def close_async_engine():
    """Close async database engine and clean up connections.

    Call this on application shutdown to gracefully close database connections.
    """
    global _async_engine, _async_session_maker

    if _async_engine is not None:
        await _async_engine.dispose()
        logger.info("Closed async database engine")
        _async_engine = None
        _async_session_maker = None
