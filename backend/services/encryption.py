"""
Encryption service for securing API keys and sensitive data.
"""

import os
import logging
from typing import Optional
from cryptography.fernet import Fernet


class EncryptionService:
    """Service for encrypting and decrypting sensitive data."""

    def __init__(self, key: Optional[str] = None):
        """Initialize encryption service.

        Args:
            key: Base64-encoded encryption key. If None, uses ROUTING_ENCRYPTION_KEY env var.
        """
        logger = logging.getLogger(__name__)

        # Treat empty strings as missing to avoid accidentally passing "" through to Fernet.
        if not key:
            key = (os.getenv("ROUTING_ENCRYPTION_KEY") or "").strip() or None

        env = (os.getenv("ENVIRONMENT") or os.getenv("environment") or "").strip().lower()
        is_production = env == "production"

        if not key:
            if is_production:
                raise ValueError("ROUTING_ENCRYPTION_KEY environment variable must be set")

            # Development-friendly behavior: generate an ephemeral key so the app can boot.
            # This is safe only when you are not trying to decrypt previously stored secrets.
            key = Fernet.generate_key().decode()
            os.environ.setdefault("ROUTING_ENCRYPTION_KEY", key)
            logger.warning(
                "ROUTING_ENCRYPTION_KEY not set; generated an ephemeral key for development."
            )

        try:
            self.cipher = Fernet(key.encode())
        except Exception as exc:
            if is_production:
                raise
            # Don't brick dev/staging due to a malformed key (common during local setup).
            key = Fernet.generate_key().decode()
            os.environ["ROUTING_ENCRYPTION_KEY"] = key
            logger.warning(
                "Invalid ROUTING_ENCRYPTION_KEY; generated an ephemeral key for development.",
                extra={"error": str(exc)},
            )
            self.cipher = Fernet(key.encode())

    def encrypt(self, data: str) -> str:
        """Encrypt a string.

        Args:
            data: String to encrypt

        Returns:
            Base64-encoded encrypted string
        """
        return self.cipher.encrypt(data.encode()).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt a string.

        Args:
            encrypted_data: Base64-encoded encrypted string

        Returns:
            Decrypted string
        """
        return self.cipher.decrypt(encrypted_data.encode()).decode()
