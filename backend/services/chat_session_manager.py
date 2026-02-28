"""
ChatSessionManager Service for managing chat sessions.

This service handles all session management for chat completion requests,
separating session concerns from the main chat handler.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class SessionMetadata:
    """Metadata for a chat session."""

    session_id: str
    created_at: float
    updated_at: float
    last_activity: float
    message_count: int
    total_tokens: int
    user_id: Optional[str]
    client_ip: Optional[str]
    session_type: str = "chat"
    tags: List[str] = None
    history: List[Dict[str, Any]] = None
    last_model: Optional[str] = None


@dataclass
class SessionMetrics:
    """Metrics for a chat session."""

    session_id: str
    request_count: int = 0
    total_tokens: int = 0
    total_response_time: float = 0.0
    average_response_time: float = 0.0
    error_count: int = 0
    success_count: int = 0
    first_request_time: Optional[float] = None
    last_request_time: Optional[float] = None


class SessionConfig:
    """Backward-compatible placeholder for session configuration used by tests."""

    def __init__(self, session_timeout_hours: int = 24, **kwargs):
        self.session_timeout_hours = session_timeout_hours
        for k, v in kwargs.items():
            setattr(self, k, v)


class SessionState:
    """Simple enum-like session states for tests."""

    ACTIVE = "active"
    EXPIRED = "expired"
    PENDING = "pending"
    CLOSED = "closed"


class ChatSessionManager:
    """Service for managing chat sessions."""

    def __init__(self, session_timeout_hours: int = 24):
        """Initialize the ChatSessionManager."""
        self.session_timeout_hours = session_timeout_hours
        self.sessions: Dict[str, SessionMetadata] = {}
        self.metrics: Dict[str, SessionMetrics] = {}

        # Cleanup tracking
        self.last_cleanup_time = time.time()

    async def get_or_create_session(
        self, session_id: str, user_id: Optional[str], client_ip: Optional[str] = None
    ) -> SessionMetadata:
        """
        Get or create a chat session.

        Args:
            session_id: Session ID
            user_id: User ID (optional)
            client_ip: Client IP address (optional)

        Returns:
            Session metadata
        """
        try:
            current_time = time.time()

            # Clean up old sessions periodically
            await self._cleanup_old_sessions()

            if session_id in self.sessions:
                # Update existing session
                session = self.sessions[session_id]
                session.updated_at = current_time
                session.last_activity = current_time

                # Update user info if provided
                if user_id:
                    session.user_id = user_id
                if client_ip:
                    session.client_ip = client_ip

                logger.debug(f"Updated existing session: {session_id}")
            else:
                # Create new session
                session = SessionMetadata(
                    session_id=session_id,
                    created_at=current_time,
                    updated_at=current_time,
                    last_activity=current_time,
                    message_count=0,
                    total_tokens=0,
                    user_id=user_id,
                    client_ip=client_ip,
                    tags=[],
                    history=[],
                    last_model=None,
                )
                self.sessions[session_id] = session

                # Initialize metrics
                self.metrics[session_id] = SessionMetrics(session_id=session_id)

                logger.info(f"Created new session: {session_id}")

            return session

        except Exception as e:
            logger.error(f"Error getting or creating session {session_id}: {e}")
            # Return a basic session metadata even if there's an error
            return SessionMetadata(
                session_id=session_id,
                created_at=time.time(),
                updated_at=time.time(),
                last_activity=time.time(),
                message_count=0,
                total_tokens=0,
                user_id=user_id,
                client_ip=client_ip,
                tags=[],
                history=[],
                last_model=None,
            )

    async def update_session_with_request(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> bool:
        """
        Update session with request data.

        Args:
            session_id: Session ID
            messages: List of messages
            model: Model name
            temperature: Temperature setting
            max_tokens: Max tokens setting

        Returns:
            True if update successful, False otherwise
        """
        try:
            await asyncio.sleep(0)  # Make this truly async for test compatibility
            current_time = time.time()

            # Get or create session
            if session_id not in self.sessions:
                session = SessionMetadata(
                    session_id=session_id,
                    created_at=current_time,
                    updated_at=current_time,
                    last_activity=current_time,
                    message_count=0,
                    total_tokens=0,
                    user_id=None,
                    client_ip=None,
                    tags=[],
                    history=[],
                    last_model=None,
                )
                self.sessions[session_id] = session
            else:
                session = self.sessions[session_id]

            # Update session activity
            session.updated_at = current_time
            session.last_activity = current_time

            # Update history and model
            if messages:
                session.history = messages
            session.last_model = model

            return True

        except Exception as e:
            logger.error(f"Error updating session {session_id} with request: {e}")
            return False

    async def get_session(self, session_id: str) -> Optional[SessionMetadata]:
        """
        Get a session by ID.

        Args:
            session_id: Session ID

        Returns:
            Session metadata if found, None otherwise
        """
        try:
            await asyncio.sleep(0)  # Make this truly async for test compatibility
            return self.sessions.get(session_id)
        except Exception as e:
            logger.error(f"Error getting session {session_id}: {e}")
            return None

    async def update_session_metadata(
        self, session_id: str, updates: Dict[str, Any]
    ) -> Optional[SessionMetadata]:
        """
        Update session metadata.

        Args:
            session_id: Session ID
            updates: Dictionary of updates to apply

        Returns:
            Updated session metadata, or None if session not found
        """
        try:
            if session_id not in self.sessions:
                logger.warning(f"Session not found: {session_id}")
                return None

            session = self.sessions[session_id]
            current_time = time.time()

            # Apply updates
            for key, value in updates.items():
                if hasattr(session, key):
                    setattr(session, key, value)

            # Update timestamps
            session.updated_at = current_time
            session.last_activity = current_time

            logger.debug(f"Updated session metadata for {session_id}: {updates}")
            return session

        except Exception as e:
            logger.error(f"Error updating session metadata for {session_id}: {e}")
            return None

    async def increment_session_counters(
        self, session_id: str, message_count: int = 1, tokens: int = 0
    ) -> Optional[SessionMetadata]:
        """
        Increment session counters.

        Args:
            session_id: Session ID
            message_count: Number of messages to add
            tokens: Number of tokens to add

        Returns:
            Updated session metadata, or None if session not found
        """
        try:
            if session_id not in self.sessions:
                logger.warning(f"Session not found: {session_id}")
                return None

            session = self.sessions[session_id]
            session.message_count += message_count
            session.total_tokens += tokens
            session.updated_at = time.time()
            session.last_activity = time.time()

            logger.debug(
                f"Incremented counters for session {session_id}: +{message_count} messages, +{tokens} tokens"
            )
            return session

        except Exception as e:
            logger.error(f"Error incrementing session counters for {session_id}: {e}")
            return None

    async def get_session_metrics(self, session_id: str) -> Optional[SessionMetrics]:
        """
        Get session metrics.

        Args:
            session_id: Session ID

        Returns:
            Session metrics, or None if session not found
        """
        try:
            if session_id not in self.metrics:
                logger.warning(f"Session metrics not found: {session_id}")
                return None

            return self.metrics[session_id]

        except Exception as e:
            logger.error(f"Error getting session metrics for {session_id}: {e}")
            return None

    async def update_session_metrics(
        self, session_id: str, metrics_update: Dict[str, Any]
    ) -> Optional[SessionMetrics]:
        """
        Update session metrics.

        Args:
            session_id: Session ID
            metrics_update: Dictionary of metrics to update

        Returns:
            Updated session metrics, or None if session not found
        """
        try:
            if session_id not in self.metrics:
                logger.warning(f"Session metrics not found: {session_id}")
                return None

            metrics = self.metrics[session_id]
            current_time = time.time()

            # Apply updates
            for key, value in metrics_update.items():
                if hasattr(metrics, key):
                    if key in ["total_response_time", "total_tokens"]:
                        # Additive updates
                        setattr(metrics, key, getattr(metrics, key) + value)
                    elif key in ["request_count", "error_count", "success_count"]:
                        # Increment updates
                        setattr(metrics, key, getattr(metrics, key) + value)
                    else:
                        # Direct assignment
                        setattr(metrics, key, value)

            # Update timestamps
            if (
                "request_count" in metrics_update
                and metrics_update["request_count"] > 0
            ):
                if not metrics.first_request_time:
                    metrics.first_request_time = current_time
                metrics.last_request_time = current_time

            # Recalculate average response time
            if metrics.request_count > 0:
                metrics.average_response_time = (
                    metrics.total_response_time / metrics.request_count
                )

            logger.debug(f"Updated session metrics for {session_id}: {metrics_update}")
            return metrics

        except Exception as e:
            logger.error(f"Error updating session metrics for {session_id}: {e}")
            return None

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session ID

        Returns:
            True if session was deleted, False if not found
        """
        try:
            if session_id in self.sessions:
                del self.sessions[session_id]
                if session_id in self.metrics:
                    del self.metrics[session_id]
                logger.info(f"Deleted session: {session_id}")
                return True
            else:
                logger.warning(f"Session not found for deletion: {session_id}")
                return False

        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False

    async def list_sessions(
        self, user_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List sessions with optional filtering.

        Args:
            user_id: Filter by user ID (optional)
            limit: Maximum number of sessions to return

        Returns:
            List of session metadata
        """
        try:
            sessions = []

            for session_id, metadata in self.sessions.items():
                # Filter by user_id if provided
                if user_id and metadata.user_id != user_id:
                    continue

                sessions.append(
                    {
                        "session_id": session_id,
                        "metadata": asdict(metadata),
                        "metrics": asdict(
                            self.metrics.get(
                                session_id, SessionMetrics(session_id=session_id)
                            )
                        ),
                    }
                )

                if len(sessions) >= limit:
                    break

            # Sort by last activity
            sessions.sort(key=lambda x: x["metadata"]["last_activity"], reverse=True)

            return sessions

        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            return []

    async def get_session_history(
        self, session_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get session history (could be extended to include actual messages).

        Args:
            session_id: Session ID
            limit: Maximum number of history items to return

        Returns:
            List of session history items
        """
        try:
            # This is a placeholder - in a real implementation, this would
            # retrieve actual message history from a database
            metadata = self.sessions.get(session_id)
            if not metadata:
                return []

            # Return basic session info as history
            return [
                {
                    "timestamp": metadata.created_at,
                    "event": "session_created",
                    "details": {
                        "session_id": session_id,
                        "user_id": metadata.user_id,
                        "client_ip": metadata.client_ip,
                        "message_count": metadata.message_count,
                        "total_tokens": metadata.total_tokens,
                    },
                }
            ]

        except Exception as e:
            logger.error(f"Error getting session history for {session_id}: {e}")
            return []

    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        try:
            current_time = time.time()
            cutoff_time = current_time - (self.session_timeout_hours * 3600)

            sessions_to_remove = []
            for session_id, metadata in self.sessions.items():
                if metadata.last_activity < cutoff_time:
                    sessions_to_remove.append(session_id)

            for session_id in sessions_to_remove:
                await self.delete_session(session_id)

            if sessions_to_remove:
                logger.info(f"Cleaned up {len(sessions_to_remove)} expired sessions")

            return len(sessions_to_remove)

        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {e}")
            return 0

    async def _cleanup_old_sessions(self) -> None:
        """Clean up old sessions periodically."""
        current_time = time.time()

        # Only cleanup every 10 minutes to avoid performance impact
        if current_time - self.last_cleanup_time < 600:
            return

        self.last_cleanup_time = current_time
        await self.cleanup_expired_sessions()

    async def get_session_stats(self) -> Dict[str, Any]:
        """Get overall session statistics."""
        try:
            current_time = time.time()
            cutoff_time = current_time - (self.session_timeout_hours * 3600)

            stats = {
                "total_sessions": len(self.sessions),
                "active_sessions": 0,
                "expired_sessions": 0,
                "total_messages": 0,
                "total_tokens": 0,
                "unique_users": set(),
                "session_timeout_hours": self.session_timeout_hours,
                "timestamp": current_time,
            }

            for session_id, metadata in self.sessions.items():
                if metadata.last_activity >= cutoff_time:
                    stats["active_sessions"] += 1
                else:
                    stats["expired_sessions"] += 1

                stats["total_messages"] += metadata.message_count
                stats["total_tokens"] += metadata.total_tokens

                if metadata.user_id:
                    stats["unique_users"].add(metadata.user_id)

            stats["unique_users"] = len(stats["unique_users"])

            return stats

        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            return {"error": str(e)}

    async def add_session_tag(self, session_id: str, tag: str) -> bool:
        """
        Add a tag to a session.

        Args:
            session_id: Session ID
            tag: Tag to add

        Returns:
            True if tag was added, False if session not found
        """
        try:
            if session_id not in self.sessions:
                return False

            session = self.sessions[session_id]
            if tag not in session.tags:
                session.tags.append(tag)
                session.updated_at = time.time()
                logger.debug(f"Added tag '{tag}' to session {session_id}")

            return True

        except Exception as e:
            logger.error(f"Error adding tag to session {session_id}: {e}")
            return False

    async def remove_session_tag(self, session_id: str, tag: str) -> bool:
        """
        Remove a tag from a session.

        Args:
            session_id: Session ID
            tag: Tag to remove

        Returns:
            True if tag was removed, False if session not found or tag not present
        """
        try:
            if session_id not in self.sessions:
                return False

            session = self.sessions[session_id]
            if tag in session.tags:
                session.tags.remove(tag)
                session.updated_at = time.time()
                logger.debug(f"Removed tag '{tag}' from session {session_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error removing tag from session {session_id}: {e}")
            return False

    async def get_sessions_by_tag(self, tag: str) -> List[str]:
        """
        Get all session IDs with a specific tag.

        Args:
            tag: Tag to search for

        Returns:
            List of session IDs
        """
        try:
            session_ids = []
            for session_id, metadata in self.sessions.items():
                if tag in metadata.tags:
                    session_ids.append(session_id)

            return session_ids

        except Exception as e:
            logger.error(f"Error getting sessions by tag '{tag}': {e}")
            return []

    async def is_session_active(self, session_id: str) -> bool:
        """
        Check if a session is currently active.

        Args:
            session_id: Session ID

        Returns:
            True if session is active, False otherwise
        """
        try:
            if session_id not in self.sessions:
                return False

            metadata = self.sessions[session_id]
            current_time = time.time()
            cutoff_time = current_time - (self.session_timeout_hours * 3600)

            return metadata.last_activity >= cutoff_time

        except Exception as e:
            logger.error(f"Error checking if session {session_id} is active: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get session management statistics.

        Returns:
            Dict containing session statistics
        """
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": len(
                [
                    s
                    for s in self.sessions.values()
                    if self.is_session_active(s.session_id)
                ]
            ),
            "expired_sessions": 0,  # Placeholder - would track actual expired sessions
            "session_operations": 0,  # Placeholder - would track operations count
            "timestamp": time.time(),
        }
